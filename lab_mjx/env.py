"""
Unitree A1 MJX Environment — GPU-accelerated MuJoCo via JAX.

Port of lab/env.py from SB3/Gymnasium → Brax PipelineEnv + MJX.

Key architectural differences vs lab/env.py:
  - All state is an immutable JAX PyTree (no mutable self.data).
  - Physics runs on GPU via mjx.step (wrapped by PipelineEnv.pipeline_step).
  - Thousands of environments run in parallel via jax.vmap inside Brax's PPO.
  - Observations, rewards, and done flags are float32 JAX arrays.

Design notes:
  - Curriculum penalty fades are set to 1.0 (fully active) from the start.
    The SB3 curriculum tracked global training timesteps across all envs via
    a callback — that pattern doesn't map cleanly to MJX's pure-functional
    paradigm where each env only sees its own per-episode step count. With
    thousands of parallel envs and much faster convergence, starting with full
    penalties is a reasonable baseline. Re-add curriculum via a training wrapper.
  - Debug visualisation arrows from lab/env.py are CPU-only and live in test.py.
  - ref_vel is fixed to [1, 0, 0] forward on reset (same as current lab/ setting).
"""

import mujoco
import jax
import jax.numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


def _filter_contacts(mj: mujoco.MjModel) -> mujoco.MjModel:
    """
    Restrict MJX contact detection to foot-geom vs floor only.

    The Unitree A1 uses detailed mesh collision geoms for every body segment.
    MJX pre-allocates efc_J of shape (ncon × condim, nv) for ALL envs in one
    batched array.  With ~100k potential mesh-mesh contact pairs, this alone
    consumes 25-35 GB for 1024 envs — more than any single GPU has.

    For locomotion we only need foot-floor contacts.  Disabling all other geoms
    drops ncon from ~100k → ~16 (4 feet × 4 max contacts each), cutting
    contact memory by >1000× and making MJX viable on a T4 / RTX 3060.

    The _is_fallen() guard (pitch/roll/height thresholds) handles fall
    detection, so the missing body-floor contacts are not a problem.
    """
    # Disable all geom collisions to start from a clean slate
    mj.geom_contype[:]     = 0
    mj.geom_conaffinity[:] = 0

    # Zero out margins and gaps as MJX does not support margin/gap for plane-mesh contacts
    mj.geom_margin[:]      = 0.0
    mj.geom_gap[:]         = 0.0

    for gid in range(mj.ngeom):
        body_id   = int(mj.geom_bodyid[gid])
        geom_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, gid)    or ''
        body_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
        combined  = (geom_name + ' ' + body_name).lower()

        is_world = (body_id == 0)          # world body = floor plane
        is_foot  = (any(kw in combined      # Unitree A1 foot/calf links
                       for kw in ('foot', 'calf'))
                    and mj.geom_group[gid] != 2) # exclude visual geoms (group 2)

        if is_world or is_foot:
            mj.geom_contype[gid]     = 1
            mj.geom_conaffinity[gid] = 1

    n_active = int((mj.geom_contype > 0).sum())
    print(f'[contact filter] {n_active}/{mj.ngeom} geoms active '
          f'({mj.ngeom - n_active} disabled)')
    return mj


class UnitreeA1MJXEnv(PipelineEnv):
    """Unitree A1 locomotion — Brax PipelineEnv backed by MJX for GPU training."""

    def __init__(self, xml_path: str, n_substeps: int = 10,
                 filter_contacts: bool = True):
        """
        Args:
            xml_path:        Path to the MuJoCo scene XML.
            n_substeps:      Physics steps per control step → 50 Hz.
            filter_contacts: If True (default), restrict collision detection
                             to foot-geom vs floor only.  This is essential
                             for GPU training — see _filter_contacts() above.
                             Set to False only if you need full body contacts
                             (e.g. for a fall-recovery task).
        """
        # Load and optionally filter the MjModel before building the brax system
        mj = mujoco.MjModel.from_xml_path(xml_path)
        if filter_contacts:
            mj = _filter_contacts(mj)
        # Override Newton solver iterations to ensure simulation stability (prevents NaN explosions)
        # Since scene.xml sets iterations="1", the solver doesn't converge, causing explosions.
        # iterations = 10 is standard and rock-solid stable for quadruped training.
        mj.opt.iterations = 10
        # brax.io.mjcf.load_model accepts a pre-built MjModel
        sys = mjcf.load_model(mj)
        super().__init__(sys, backend='mjx', n_frames=n_substeps)

        # Load plain MjModel for metadata (nq, nv, nu, keyframes)
        mj = mujoco.MjModel.from_xml_path(xml_path)
        self._nq = mj.nq
        self._nv = mj.nv
        self._nu = mj.nu

        # Keyframe 0 qpos as starting pose (same as mj_resetDataKeyframe(model, data, 0))
        if mj.nkey > 0:
            self._init_qpos = jp.array(mj.key_qpos[0], dtype=jp.float32)
        else:
            self._init_qpos = jp.array(mj.qpos0, dtype=jp.float32)
        self._init_qvel = jp.zeros(mj.nv, dtype=jp.float32)

        # Joint configuration (must match lab/env.py exactly)
        self._default_dof_pos = jp.array([
            -0.1, 0.8, -1.5,   # Front Right (hip, thigh, calf)
             0.1, 0.8, -1.5,   # Front Left
            -0.1, 1.0, -1.5,   # Rear Right
             0.1, 1.0, -1.5,   # Rear Left
        ], dtype=jp.float32)
        self._action_scale = jp.float32(0.8)   # Max deviation from default pose
        self._skew_alpha   = jp.float32(0.1)   # EMA decay for running pitch/roll

        # Fall thresholds (matching lab/env.py)
        self._fall_angle = jp.float32(25.0 * jp.pi / 180.0)  # 25 degrees
        self._min_height = jp.float32(0.18)
        self._max_height = jp.float32(0.50)

    # ── Brax required properties ──────────────────────────────────────

    @property
    def observation_size(self) -> int:
        """50-dim observation (identical to lab/env.py):
        base_lin_vel(3) + base_ang_vel(3) + orientation(2) +
        joint_pos(12) + joint_vel(12) + last_action(12) +
        running_skew(2) + ref_vel(3) + ref_height(1) = 50
        """
        return 50

    @property
    def action_size(self) -> int:
        return self._nu  # 12

    # ── Brax interface ────────────────────────────────────────────────

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng_pos, rng_vel, rng_height = jax.random.split(rng, 4)

        # Small random perturbation around keyframe pose (same as lab/env.py)
        qpos = self._init_qpos + jax.random.uniform(
            rng_pos, (self._nq,), minval=-0.01, maxval=0.01)
        qvel = self._init_qvel + jax.random.uniform(
            rng_vel, (self._nv,), minval=-0.01, maxval=0.01)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # Commands (same distribution as lab/env.py reset)
        ref_vel    = jp.array([1.0, 0.0, 0.0], dtype=jp.float32)  # fixed forward
        ref_height = jax.random.uniform(rng_height, (), minval=-1.0, maxval=1.0)

        last_action      = jp.zeros(self._nu, dtype=jp.float32)
        last_last_action = jp.zeros(self._nu, dtype=jp.float32)
        running_pitch    = jp.zeros((), dtype=jp.float32)
        running_roll     = jp.zeros((), dtype=jp.float32)

        obs = self._get_obs(
            pipeline_state, last_action,
            running_pitch, running_roll,
            ref_vel, ref_height,
        )

        zero = jp.zeros((), dtype=jp.float32)

        # metrics: Brax PPO automatically averages these across eval episodes
        # and passes them to progress_fn → WandB
        metrics = {
            'reward/velocity':               zero,
            'reward/angular_velocity':       zero,
            'reward/height':                 zero,
            'penalty/hip_similarity':        zero,
            'penalty/pose_similarity':       zero,
            'penalty/action_2nd_derivative': zero,
            'penalty/vertical_velocity':     zero,
            'penalty/a_pitch_error':         zero,
            'penalty/a_roll_error':          zero,
            'penalty/symmetry':              zero,
            'penalty/fall':                  zero,
        }

        # info: persistent state across steps within an episode
        info = {
            'ref_vel':          ref_vel,
            'ref_height':       ref_height,
            'last_action':      last_action,
            'last_last_action': last_last_action,
            'running_pitch':    running_pitch,
            'running_roll':     running_roll,
        }

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=zero,
            done=zero,
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        action = jp.clip(action, -1.0, 1.0)

        # Map [-1,1] action to absolute position targets (identical to lab/env.py)
        ctrl = self._default_dof_pos + action * self._action_scale
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        yaw, pitch, roll = self._get_euler(pipeline_state.qpos)

        # Exponential moving average for running orientation (same α as lab/env.py)
        running_pitch = ((1.0 - self._skew_alpha) * state.info['running_pitch']
                        + self._skew_alpha * pitch)
        running_roll  = ((1.0 - self._skew_alpha) * state.info['running_roll']
                        + self._skew_alpha * roll)

        ref_vel    = state.info['ref_vel']
        ref_height = state.info['ref_height']

        obs = self._get_obs(
            pipeline_state, action,
            running_pitch, running_roll,
            ref_vel, ref_height,
        )

        reward, components = self._compute_reward(
            pipeline_state, action,
            state.info['last_action'],
            state.info['last_last_action'],
            running_pitch, running_roll,
            ref_vel, ref_height, yaw,
        )

        done = self._is_fallen(pipeline_state)

        info = {
            **state.info,
            'last_action':      action,
            'last_last_action': state.info['last_action'],
            'running_pitch':    running_pitch,
            'running_roll':     running_roll,
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={**state.metrics, **components},  # preserve wrapper-added keys (like 'reward')
            info=info,
        )

    # ── Internal helpers (all JAX-compatible / JIT-safe) ─────────────

    def _get_euler(self, qpos: jp.ndarray):
        """yaw, pitch, roll from base quaternion qpos[3:7] = (w, x, y, z)."""
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw   = jp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        pitch = jp.arcsin(jp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        roll  = jp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        return yaw, pitch, roll

    def _get_obs(
        self,
        pipeline_state,
        last_action:   jp.ndarray,
        running_pitch: jp.ndarray,
        running_roll:  jp.ndarray,
        ref_vel:       jp.ndarray,
        ref_height:    jp.ndarray,
    ) -> jp.ndarray:
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        _, pitch, roll = self._get_euler(qpos)

        return jp.concatenate([
            qvel[0:3] / 3.0,                                          # base linear vel   (3)
            qvel[3:6] / 2.0,                                          # base angular vel  (3)
            jp.array([roll, pitch]) / jp.pi,                          # orientation       (2)
            (qpos[7:] - self._default_dof_pos) / self._action_scale,  # joint pos delta   (12)
            qvel[6:] / 21.0,                                          # joint vel         (12)
            last_action,                                              # last action       (12)
            jp.array([running_pitch, running_roll]),                   # running skew EMA  (2)
            ref_vel,                                                  # cmd velocity      (3)
            jp.array([ref_height]),                                    # cmd height        (1)
        ])  # total = 50

    def _gaus(self, x: jp.ndarray, *alphas: float) -> jp.ndarray:
        """Gaussian kernel(s) evaluated at x, averaged if multiple alphas given."""
        return jp.mean(jp.stack([jp.exp(-jp.float32(a) * x**2) for a in alphas]))

    def _compute_reward(
        self,
        pipeline_state,
        action:           jp.ndarray,
        last_action:      jp.ndarray,
        last_last_action: jp.ndarray,
        running_pitch:    jp.ndarray,
        running_roll:     jp.ndarray,
        ref_vel:          jp.ndarray,
        ref_height:       jp.ndarray,
        yaw:              jp.ndarray,
    ):
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel

        # ── Velocity tracking ─────────────────────────────────────────
        world_vel = qvel[0:2]
        cos_y, sin_y = jp.cos(-yaw), jp.sin(-yaw)
        # Rotate world velocity into robot frame
        actual_vel = jp.concatenate([
            (cos_y * world_vel[0] - sin_y * world_vel[1])[None],
            (sin_y * world_vel[0] + cos_y * world_vel[1])[None],
        ])
        ref_vel_s  = jp.array([5.0 * ref_vel[0], 2.5 * ref_vel[1]])
        actual_spd = jp.linalg.norm(actual_vel)
        ref_spd    = jp.linalg.norm(ref_vel_s)

        cos_sim = jp.where(
            ref_spd > 1e-4,
            jp.where(actual_spd > 1e-4,
                     jp.dot(actual_vel, ref_vel_s) / (actual_spd * ref_spd + 1e-8),
                     jp.float32(0.0)),
            jp.float32(1.0),
        )
        speed_reward       = self._gaus(actual_spd - ref_spd, 4.0)
        angular_vel_reward = self._gaus(qvel[5] - ref_vel[2], 100.0, 10.0, 1.0)

        # ── Height tracking ───────────────────────────────────────────
        ref_h         = ref_height * jp.float32(0.08) + jp.float32(0.28)
        height_reward = self._gaus(qpos[2] - ref_h, 100.0)

        # ── Stability penalties (fades = 1.0; see module docstring) ───
        hip_idx  = jp.array([0, 3, 6, 9])
        hip_sim  = -jp.sum(jp.square(qpos[7 + hip_idx] - self._default_dof_pos[hip_idx]))
        pose_sim = -jp.sum(jp.square(qpos[7:] - self._default_dof_pos))
        accel_p  = -jp.sum(jp.square(action - 2.0 * last_action + last_last_action))
        vert_vel = -qvel[2] ** 2
        pitch_p  = -running_pitch ** 2
        roll_p   = -running_roll  ** 2

        # ── Symmetry (diagonal pairs should mirror each other) ────────
        q = qpos[7:19]
        # Negate hip joint (index 0 within each 3-DOF leg block) for mirroring
        q_FR = jp.concatenate([-q[0:1], q[1:3]])   # Front Right, hip negated
        q_FL = q[3:6]                               # Front Left
        q_RR = jp.concatenate([-q[6:7], q[7:9]])   # Rear Right, hip negated
        q_RL = q[9:12]                              # Rear Left
        sym_p = -(jp.sum(jp.square(q_FR - q_RL)) + jp.sum(jp.square(q_FL - q_RR)))

        fall_p = jp.where(self._is_fallen(pipeline_state) > 0.5,
                         jp.float32(-1.0), jp.float32(0.0))

        # ── Combine (weights identical to lab/env.py) ─────────────────
        penalty_sum = (
            1.000 * hip_sim  +
            0.020 * pose_sim +
            0.250 * vert_vel +
            1.000 * pitch_p  +
            1.000 * roll_p   +
            0.500 * sym_p    +
            0.005 * accel_p
        )
        total = jp.float32(0.8) * cos_sim * speed_reward + jp.float32(0.2) * angular_vel_reward

        components = {
            'reward/velocity':               cos_sim * speed_reward,
            'reward/angular_velocity':       angular_vel_reward,
            'reward/height':                 height_reward,
            'penalty/hip_similarity':        hip_sim,
            'penalty/pose_similarity':       pose_sim,
            'penalty/action_2nd_derivative': accel_p,
            'penalty/vertical_velocity':     vert_vel,
            'penalty/a_pitch_error':         pitch_p,
            'penalty/a_roll_error':          roll_p,
            'penalty/symmetry':              sym_p,
            'penalty/fall':                  fall_p,
        }
        return total + penalty_sum + fall_p, components

    def _is_fallen(self, pipeline_state) -> jp.ndarray:
        """Returns 1.0 if the robot has fallen, 0.0 otherwise (float32 for Brax)."""
        _, pitch, roll = self._get_euler(pipeline_state.qpos)
        z = pipeline_state.qpos[2]
        bad_ori = (jp.abs(pitch) > self._fall_angle) | (jp.abs(roll) > self._fall_angle)
        bad_z   = (z < self._min_height) | (z > self._max_height)
        return (bad_ori | bad_z).astype(jp.float32)
