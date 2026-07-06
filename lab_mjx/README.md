# lab_mjx — GPU-Accelerated Training with MJX + Brax

Port of the `lab/` SB3/MuJoCo training to **MuJoCo MJX** (JAX-native physics) +
**Brax PPO** for massively parallel GPU training.

## Why MJX?

| | `lab/` (SB3 + CPU) | `lab_mjx/` (MJX + GPU) |
|---|---|---|
| Parallel envs | 8 (SubprocVecEnv, CPU) | 2048+ (jax.vmap, GPU) |
| 100M step time | ~6-8 hours | ~15-30 minutes |
| Memory model | Mutable Python objects | Immutable JAX PyTrees |
| Replay/reuse | pickle checkpoints | pickle + Brax orbax |

## Setup

> [!IMPORTANT]
> **JAX has no native Windows CUDA wheels.** The `jax[cuda12]` package does not
> ship GPU-enabled `jaxlib` for Windows on PyPI — this is an ongoing JAX limitation.
> **WSL2 (Windows Subsystem for Linux) is the only supported path for GPU training.**
>
> The `test.py` inference script works fine on native Windows (CPU-only JAX is fine
> for a single inference pass). Only `train.py` needs the GPU stack.

### Option A — WSL2 (Recommended for training)

```bash
# 1. Install WSL2 + Ubuntu (one-time, in PowerShell as Admin):
wsl --install -d Ubuntu-24.04

# 2. Inside WSL2, clone / mount the project and create a venv:
cd /mnt/d/Files/Scripts/py/Graduation\ Project

python3 -m venv .venv_wsl
source .venv_wsl/bin/activate

# 3. Install JAX with CUDA 12 (works inside WSL2 — CUDA is shared with Windows):
pip install "jax[cuda12]"
python -c "import jax; print(jax.devices())"   # → [CudaDevice(id=0)]

# 4. Install the rest:
pip install mujoco mujoco-mjx brax flax optax wandb mediapy

# 5. Train:
python lab_mjx/train.py
```

WSL2 sees the same NVIDIA GPU via the CUDA-WSL2 driver — no separate Linux driver needed
as long as your Windows NVIDIA driver is ≥ 470.x.

### Option B — Native Windows (CPU only, for testing/inference)

The existing `.venv` works for running `test.py` after loading a checkpoint trained in WSL2.
`brax` and `mujoco-mjx` need to be installed for the CPU backend:

```powershell
# From the project root (PowerShell):
.venv\Scripts\pip.exe install mujoco-mjx brax flax optax wandb mediapy

# Verify:
.venv\Scripts\python.exe -c "import brax; print('brax OK')"
```

Training on CPU is extremely slow (minutes per iteration vs seconds on GPU),
but the code will run for quick sanity checks.

### 3. Run from the project root

```bash
# Training (inside WSL2 venv):
cd /mnt/d/Files/Scripts/py/Graduation\ Project
source .venv_wsl/bin/activate
python lab_mjx/train.py

# Inference + MuJoCo viewer (native Windows, after training):
cd "d:\Files\Scripts\py\Graduation Project"
.venv\Scripts\python.exe lab_mjx/test.py --version 1.0

# Custom velocity commands:
.venv\Scripts\python.exe lab_mjx/test.py --version 1.0 --ref-vx 2.0 --ref-wz 0.5 --speed 0.5
```

## File Structure

```
lab_mjx/
├── env.py          # UnitreeA1MJXEnv — Brax PipelineEnv subclass
├── train.py        # Brax PPO training with WandB + checkpointing
├── test.py         # CPU mujoco.viewer inference (loads trained params)
├── requirements.txt
└── README.md
```

## Key Differences from `lab/`

### Environment (`env.py`)

- Inherits `brax.envs.base.PipelineEnv` instead of `gymnasium.Env`
- `reset(rng)` and `step(state, action)` return immutable `State` PyTrees
- All NumPy → JAX arrays (`jax.numpy`)
- Physics via `self.pipeline_step(state, ctrl)` → `mjx.step` on GPU
- Per-episode state (`last_action`, `running_pitch/roll`) stored in `state.info`
- Reward components in `state.metrics` (auto-averaged by Brax for logging)

### Curriculum

The SB3 curriculum (gradually introducing penalties via a callback) is **not active**
in this port. All penalty fades start at **1.0** (fully active). This is because the
pure-functional MJX paradigm doesn't naturally track global training timesteps inside
the env. With 2048 parallel environments, convergence is fast enough that progressive
curriculum is less critical. To re-add curriculum, wrap the environment with a custom
`AutoResetWrapper` that injects total env steps into `state.info`.

### Training (`train.py`)

- `brax.training.agents.ppo.train(environment, num_timesteps, ...)` — no manual
  rollout loop needed
- `num_envs=2048` environments run in parallel on the GPU via `jax.vmap`
- `progress_fn` callback logs to WandB at each evaluation checkpoint
- Saves params as `Models/a1_mjx_v{VERSION}_params.pkl` (pickle) for inference
- Also saves Brax orbax checkpoints to `Models/checkpoints_mjx/` for resuming

### Inference (`test.py`)

- Loads pickle params, recreates the JAX policy network
- Observation computed from CPU `mujoco.MjData` (same formula as `env.py`)
- Action mapped back to position targets, applied to CPU MuJoCo simulation
- Real-time rendering via `mujoco.viewer.launch_passive`

## VRAM Usage (RTX 3060 Laptop, 6 GB)

| `num_envs` | Estimated VRAM | Status |
|---|---|---|
| 1024 | ~2-3 GB | ✅ Safe |
| 2048 | ~3-4 GB | ✅ Default |
| 4096 | ~5-6 GB | ⚠️ May OOM |

If you get CUDA out-of-memory errors, reduce `NUM_ENVS` in `train.py`.

## Resuming Training

Brax saves orbax checkpoints to `Models/checkpoints_mjx/v{VERSION}/`.
To resume, add `restore_checkpoint_path=str(ckpt_dir)` to the `ppo.train()` call
in `train.py`.
