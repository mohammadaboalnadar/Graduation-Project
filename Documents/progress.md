# [v1.0](../Models/a1_walk_v1.0.zip)
- simple training loop with a basic reward function
- reward is based solely on forward speed
- trained for `5m` steps over `4` envs

Model v1.0 developed a cheating strategy where it would jump forwards to maximize forward velocity at the expense of falling over

![training metrics](../Figures/v1_progress_5m_steps.png)

reward function:
```python
reward = (
	+ 2.0 * forward_vel
	- 0.5 * lateral_vel
	- 0.01 * energy # (energy = np.sum(np.square(action)))
	+ alive_bonus
	- fall_penalty
)
```

demo video:
<video controls src="../Videos/v1.0-unnamed.mp4" title="v1.0"></video>

# [v2.0](../Models/a1_walk_v2.0.zip)
- Added a target velocity and a penalty for deviating from it instead of just rewarding forward velocity (gaussian function centered at target velocity)
- Increased the penalty for falling significantly (5 -> 50)
- Bonus reward for staying upright
- Penalty for high base angular velocity
- Trained for `10m` steps over `8` envs

Model v2.0 developed a microstepping gait that allows it to maintain a stable speed while keeping its base upright, effective but not realistic nor practical for real-world deployment

![training metrics](../Figures/v2.0_progress_10m_steps_test.png)

demo video:
<video controls src="../Videos/v2.0-unnamed.mp4" title="v2.0"></video>

# [v2.1](../Models/a1_walk_v2.1.zip)
- Increased the episode maximum steps from `1000` to `50,000` to allow the model to learn longer-term strategies and behaviors
- Trained for `30m` steps over `8` envs

Model v2.1 improved substantially over v2.0 but eventually plummeted in performance in favor of a strategy where it stays dormant and doesn't move at all, likely to maximize the alive bonus while avoiding any penalties for movement.

![training metrics](<../Figures/v2.1 - Iteration 1.png>)

demo video:
<video controls src="../Videos/v2.1-unnamed.mp4" title="v2.1"></video>