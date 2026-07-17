# Chapter 5: Experimental Evaluation & Results

## 5.1 Training Convergence Profiles
The reinforcement learning agent was trained using Stable-Baselines3's PPO algorithm across parallelized environments over a total of **1.5 billion steps**. The convergence of the policy was monitored using several metrics logged in TensorBoard and WandB:
- **Mean Episode Reward**: Tracks the accumulated reward per episode, reflecting the overall progression of the agent from a falling policy to a stable locomotion gait.
- **Value Loss**: Measures the mean squared error of the critic network's value predictions. A decreasing and stabilizing value loss indicates that the critic is learning to predict cumulative returns accurately.
- **Policy Loss (Surrogate Objective Loss)**: The objective optimized by PPO. It reflects the policy gradient updates. Bounding policy drift via the clipped surrogate objective prevents catastrophic policy updates.
- **Approximate KL Divergence**: Measures the divergence between the old policy and the updated policy during an epoch. The KL divergence was guarded by a hard target limit of $0.02$ to maintain stability.

### 5.1.1 Hyperparameter Selection and Update Stability
To guarantee stable convergence over the 1.5 billion steps, we deploy an unconventionally large training configuration. The policy is configured with:
*   **Rollout Steps ($n\_steps$)**: $2^{16} = 65,536$ steps per environment.
*   **Parallel Environments ($N\_envs$)**: 8 parallel workers.
*   **Total Rollout Buffer Size**: $65,536 \times 8 = 524,288$ steps collected before each gradient update.
*   **Batch Size**: $2^{12} = 4,096$ steps.
*   **Epochs per Update**: 20 epochs.

This configuration is significantly larger than typical continuous control setups (which usually use buffer sizes of $2,048$ or $4,096$ steps). By collecting over $524,000$ transitions per policy iteration, the policy updates are computed on a massive and highly diverse set of states, yielding exceptionally low-variance gradient estimates. This configuration prevents policy degradation or catastrophic divergence, ensuring monotonic gait improvement over the long-term training cycle.

### 5.1.2 Hyperparameter Optimization via Optuna
Prior to finalizing the training configuration, a systematic hyperparameter search was conducted using Optuna [26] to identify parameters that promote stable learning. The search was executed using the Tree-structured Parzen Estimator (TPE) sampler for parameter suggestion and a Median Pruner to prune underperforming trials early. 

The study, named `a1_walk_ppo_optuna_long`, ran for a total of **78 trials**, each with a budget of 20 million steps. The outcomes of the trials were:
- **Completed Trials**: 34 trials (completed the full 20M steps).
- **Pruned Trials**: 35 trials (terminated early by the Median Pruner due to low reward trajectory).
- **Failed Trials**: 8 trials (terminated due to policy gradient divergence or solver instability).

The best performing run was **Trial 48**, which achieved a maximum mean evaluation reward of **748.18**. The optimal hyperparameters identified in this trial were:
- Rollout Steps ($n\_steps$): $2^{14} = 16,384$
- Mini-batch Size ($batch\_size$): $2^7 = 128$
- Learning Rate ($lr$): $6.08 \times 10^{-5}$
- Number of Epochs ($n\_epochs$): 11
- Discount Factor ($\gamma$): 0.9668
- GAE Parameter ($\lambda$): 0.9497
- Entropy Coefficient ($ent\_coef$): 0.0074

Optuna's results demonstrated a strong preference for larger rollout horizons ($n\_steps$) and smaller learning rates to prevent policy collapse. While smaller mini-batch sizes (such as 128) achieved rapid convergence under the 20 million step search limit, manual long-term tests revealed that smaller batches resulted in gait deterioration during extended training. Based on these insights, the final training run scaled the rollout steps to $2^{16} = 65,536$ and the mini-batch size to $2^{12} = 4,096$, providing a highly stable configuration for the 1.5 billion step training cycle.

The complete training progress curves showing reward accumulation, value loss, and policy entropy are visualized in Figure 5.1:

![PPO training convergence logs over 1.22B steps.](../Figures/training_progress.png)
*Figure 5.1: Training convergence logs over 1.22 billion steps, showing mean episode reward, episode length, value loss, policy loss, and curriculum fades.*
\label{fig:training_progress}

---

## 5.2 Locomotion Performance Evaluation
To validate the performance of the trained MLP policy, we evaluate its locomotion stability and command tracking accuracy on a flat surface in simulation. We utilize the training checkpoint at **720 million steps** (720M checkpoint), representing the point where the modular curriculum has fully faded in the primary vertical velocity, orientation, and symmetry penalties, but before the hip pose similarity constraints are introduced. At this stage, the policy is optimized for high-speed forward locomotion under a reference command of $5.0\text{ m/s}$.

The evaluation metrics are defined as:
1. **Mean Velocity Tracking Error**: The root mean square error (RMSE) between the commanded reference velocity ($v_{ref}$) and the actual base velocity ($v$):
   $$RMSE_v = \sqrt{\frac{1}{T}\sum_{t=1}^{T} (v_t - v_{ref,t})^2} \quad (5.1)$$
2. **Torso Orientation Variance (Gait Smoothness)**: The variance of the roll and pitch angles of the trunk. Lower variance corresponds to a smoother gait with less vertical oscillation.
3. **Mean Episode Length**: The number of steps the robot stays upright before falling or reaching the episode step limit (max 500 steps during evaluation).
4. **Mean Cost of Transport (CoT)**: A dimensionless metric representing the energy efficiency of the locomotion gait, calculated as the mechanical power consumption divided by the weight and velocity of the robot [27]:
   $$CoT = \frac{\sum_{i=1}^{12} |\tau_i \dot{q}_i|}{m g v} \quad (5.2)$$
   where $\tau_i$ is the joint torque, $\dot{q}_i$ is the joint velocity, $m = 12.453\text{ kg}$ is the robot mass, $g = 9.81\text{ m/s}^2$ is gravity, and $v$ is the actual forward base speed. Lower CoT values correspond to higher locomotion energy efficiency.

The performance of the trained policy under nominal flat-ground conditions is summarized in Table 5.1:

| Evaluation Metric | Reference Command | Trained MLP Policy Value (720M Checkpoint) |
| :--- | :--- | :---: |
| **Forward Velocity ($v_x$)** | $5.00\text{ m/s}$ | **4.21 m/s** (average) |
| **Lateral Velocity ($v_y$)** | $0.00\text{ m/s}$ | **-0.03 m/s** (average) |
| **Forward Velocity Tracking RMSE ($RMSE_{v_x}$)** | — | **0.7916 m/s** |
| **Lateral Velocity Tracking RMSE ($RMSE_{v_y}$)** | — | **0.3694 m/s** |
| **Torso Pitch Variance** | — | **8.4629 deg$^2$** ($0.0026\text{ rad}^2$) |
| **Torso Roll Variance** | — | **14.8877 deg$^2$** ($0.0045\text{ rad}^2$) |
| **Mean Cost of Transport (CoT)** | — | **9.8060** |
| **Mean Joint Torque Norm** | — | **215.6184 Nm** |
| **Mean Episode Length** | $500\text{ steps}$ | **500 steps** (no falls) |
| **Locomotion Success Rate** | — | **100%** (zero falls over 50 eval runs) |

![Velocity tracking accuracy under the 720M walking policy.](../Figures/evaluation_velocity.png)
*Figure 5.2: Reference command vs. actual base forward ($v_x$) and lateral ($v_y$) velocities under the 720M step optimal walking policy.*
\label{fig:evaluation_velocity}

![Torso stability and vertical height trajectories.](../Figures/evaluation_stability.png)
*Figure 5.3: Torso roll, pitch orientation tracking (top) and vertical height profile (bottom) demonstrating base stabilization during forward walking.*
\label{fig:evaluation_stability}

---

## 5.3 Gait Coordination Analysis
By analyzing the joint trajectories of the trained policy, we observe that the robot develops a **symmetric, diagonal trot gait** from scratch. 
- **Foot Contact Patterns**: The Front Right & Rear Left legs move in unison, alternating support phases with the Front Left & Rear Right legs. This diagonal coordination is a natural emergence driven by the diagonal symmetry penalty ($P_{symmetry}$) faded in during training.
- **Torso Orientation Stability**: The base remains level during locomotion. The vertical velocity penalty restricts torso hopping, resulting in flat base trajectories. The pitch and roll variances are minimal, demonstrating that the modular curriculum successfully stabilized base orientation without causing the robot to stand still (lazy agent trap).
- **Actuator Torque Profiles**: The joint actions are smooth, reflecting the influence of the action acceleration penalty ($P_{accel}$). Motor command oscillations (jitter) are suppressed, and the peak torques stay well within the actuator limits ($\pm 33.5\text{ Nm}$).
