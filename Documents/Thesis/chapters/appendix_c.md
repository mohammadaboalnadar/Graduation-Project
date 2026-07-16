# Appendix C: Supplementary Plots and Trial Data

This appendix contains supplementary evaluation plots and statistical tables detailing the automated hyperparameter search results.

### C.1 Top 5 Optuna Study Trials
Table C.1 lists the hyperparameter configurations and final mean evaluation rewards for the top 5 completed trials of the `a1_walk_ppo_optuna_long` study, trained over 20 million steps per trial.

#### Table C.1: Top 5 Completed Optuna Trials

| Metric / Parameter | Trial 48 (Best) | Trial 55 | Trial 20 | Trial 43 | Trial 25 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Reward Value** | **748.18** | 687.58 | 686.64 | 659.68 | 659.05 |
| **Rollout Steps ($n\_steps$)** | $16,384$ ($2^{14}$) | $8,192$ ($2^{13}$) | $8,192$ ($2^{13}$) | $16,384$ ($2^{14}$) | $16,384$ ($2^{14}$) |
| **Mini-batch Size ($batch\_size$)** | 128 ($2^7$) | 64 ($2^6$) | 64 ($2^6$) | 64 ($2^6$) | 64 ($2^6$) |
| **Number of Epochs ($n\_epochs$)** | 11 | 11 | 9 | 11 | 8 |
| **Learning Rate ($lr$)** | $6.08 \times 10^{-5}$ | $3.85 \times 10^{-5}$ | $4.20 \times 10^{-5}$ | $1.04 \times 10^{-4}$ | $6.59 \times 10^{-5}$ |
| **Discount Factor ($\gamma$)** | 0.9668 | 0.9604 | 0.9704 | 0.9643 | 0.9601 |
| **GAE Parameter ($\lambda$)** | 0.9497 | 0.9216 | 0.9601 | 0.9465 | 0.9478 |
| **Entropy Coeff. ($ent\_coef$)** | $0.0074$ | $0.0083$ | $2.96 \times 10^{-5}$ | $0.0040$ | $0.0094$ |

---

### C.2 Supplementary Plots
The following figures represent copies of the training convergence and evaluation logs generated during local validation. Since these plots are critical to the results discussion, they are also embedded directly within Chapter 5.

#### Figure C.1: Training Convergence Profiles
This multi-panel figure traces the progression of PPO metrics (mean episode reward, mean episode length, policy loss, and value loss) alongside the modular curriculum scheduling fades over the 1.22 billion steps training run of the `v19.5_2` policy.

*Note: The generated image is saved locally under [Figures/training_progress.png](file:///d:/Files/Scripts/py/Graduation%20Project/Figures/training_progress.png).*

#### Figure C.2: Velocity Tracking Accuracy
This plot traces the commanded forward ($v_{ref, x} = 5.0\text{ m/s}$) and lateral ($v_{ref, y} = 0.0\text{ m/s}$) velocities against the actual base velocities in the robot's local frame. The agent tracks the lateral command with high precision ($RMSE_{v_y} = 0.37\text{ m/s}$), and tracks the forward velocity command with a stable steady-state average of $4.21\text{ m/s}$ ($RMSE_{v_x} = 0.79\text{ m/s}$).

*Note: The generated image is saved locally under [Figures/evaluation_velocity.png](file:///d:/Files/Scripts/py/Graduation%20Project/Figures/evaluation_velocity.png).*

#### Figure C.3: Torso Stability and Height Attenuation
This figure traces the torso pitch and roll orientation (in degrees) and the base height (in meters) over the course of a 10-second simulation:
*   **Orientation (Top)**: Demonstrates that the pitch and roll oscillations are kept within a tight safe envelope ($[-10^\circ, +10^\circ]$), representing the low-pass filtering effect of the running skew orientation penalty.
*   **Trunk Height (Bottom)**: Traces the vertical position of the torso, showing that it converges closely to the target height of $0.28\text{ m}$ with minimal vertical oscillation.

*Note: The generated image is saved locally under [Figures/evaluation_stability.png](file:///d:/Files/Scripts/py/Graduation%20Project/Figures/evaluation_stability.png).*
