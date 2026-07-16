# Appendix B: Key Code Implementations

This appendix contains key code implementations from the training environment and schedules. The complete public repository containing all code, configurations, and checkpoint records is available on GitHub at: [https://github.com/mohammadaboalnadar/Graduation-Project](https://github.com/mohammadaboalnadar/Graduation-Project).

### B.1 Observation Vector Formatting
The 50-dimensional observation vector is constructed in `lab/env.py` to provide the policy network with proprioceptive measurements, command states, and historical data:

```python
def _get_obs(self) -> np.ndarray:
    joint_pos           = self.data.qpos[7:].copy()
    joint_vel           = self.data.qvel[6:].copy()
    raw, pitch, roll    = self._get_euler()
    base_ang_vel        = self.data.qvel[3:6].copy()
    base_lin_vel        = self.data.qvel[0:3].copy()
    running_skew        = np.array([self.running_pitch, self.running_roll], dtype=np.float32)

    return np.concatenate([
        base_lin_vel / 3.0, # normalize to ~[-1, 1] range based on expected max speeds (3 dim)
        base_ang_vel / 2.0, # normalize to ~[-1, 1] range based on expected max speeds (3 dim)
        np.array([roll, pitch], dtype=np.float32) / np.pi,  # normalize angles to [-1, 1] (2 dim)
        (joint_pos - self.default_dof_pos) / self.action_scale,  # deviation from default standing pose, normalized to [-1, 1] (12 dim)
        joint_vel / 21.0, # A1 max joint speed is ~21 rad/s, normalizes to ~[-1, 1] (12 dim)
        self.last_action, # action command history, already in [-1, 1] (12 dim)
        running_skew,     # low-pass filtered running pitch and roll (2 dim)
        self.ref_vel,     # target command vector [vx, vy, wz] (3 dim)
        np.array([self.ref_height], dtype=np.float32), # target body height cmd (1 dim)
    ]).astype(np.float32) # Total = 50 dimensions
```

### B.2 Modular Curriculum Callback
The `ModularCurriculumCallback` class in `lab/train.py` manages the linear interpolation and IPC broadcasting of the curriculum penalty fades over the course of the training run:

```python
class ModularCurriculumCallback(BaseCallback):
    def __init__(self, schedules: dict, update_freq: int = 10_000, verbose=0):
        super().__init__(verbose)
        self.schedules = schedules
        self.update_freq = update_freq
        self.current_fades = {k: 0.0 for k in schedules.keys()}

    def _on_step(self) -> bool:
        # Throttle IPC overhead: Only broadcast every N steps
        if self.num_timesteps % self.update_freq != 0:
            return True

        step = self.num_timesteps

        for key, bounds in self.schedules.items():
            start, end = bounds["start"], bounds["end"]
            
            # Calculate linear interpolation
            if step <= start:
                fade = 0.0
            elif step >= end:
                fade = 1.0
            else:
                fade = (step - start) / (end - start)
                
            self.current_fades[key] = fade
            
            # Log to TensorBoard so you can see the curves
            self.logger.record(f"curriculum/{key}_fade", fade)

        # Broadcast the updated dictionary to all isolated environments
        self.training_env.env_method("set_penalty_fades", self.current_fades)
        
        return True
```
