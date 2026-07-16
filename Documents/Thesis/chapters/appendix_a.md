# Appendix A: Training Configuration and Hyperparameters

This appendix compiles the hyperparameter configurations used throughout the project. Table A.1 presents a side-by-side comparison of the default Stable-Baselines3 (SB3) parameters, the optimal parameters identified during the automated Optuna search (Trial 48), and the final scaled parameters settled on through manual tuning for long-term training stability.

### Table A.1: PPO Hyperparameter Comparison

| Hyperparameter | SB3 Default Value | Optuna Best (Trial 48) | Final Selection |
| :--- | :---: | :---: | :---: |
| **Number of Environments ($N_{envs}$)** | 1 | 8 | 8 |
| **Rollout Step Horizon ($n\_steps$)** | 2,048 | 16,384 ($2^{14}$) | **65,536** ($2^{16}$) |
| **Mini-batch Size ($batch\_size$)** | 64 | 128 ($2^7$) | **4,096** ($2^{12}$) |
| **Number of Epochs ($n\_epochs$)** | 10 | 11 | **20** |
| **Learning Rate ($lr$)** | $3 \times 10^{-4}$ | $6.08 \times 10^{-5}$ | **$2 \times 10^{-4}$** |
| **Discount Factor ($\gamma$)** | 0.99 | 0.9668 | **0.99** |
| **GAE Parameter ($\lambda$)** | 0.95 | 0.9497 | **0.95** |
| **Entropy Coefficient ($ent\_coef$)** | 0.0 | 0.0074 | **0.01** |
| **Value Function Coeff. ($vf\_coef$)** | 0.5 | 0.5 | **0.5** |
| **Max Gradient Norm ($max\_grad\_norm$)** | 0.5 | 0.5 | **0.5** |
| **Target KL Divergence ($target\_kl$)** | None | 0.02 | **0.02** |
| **State representation network (MLP)** | [64, 64] | [256, 256] | **[256, 256]** |
| **Activation Function** | `tanh` | `tanh` | **`tanh`** |

### Hyperparameter Scaling Rationale
The transition from Optuna's optimal parameters to the final selected parameters was dictated by the duration of the training runs. The Optuna study evaluated trials over a relatively short budget of 20 million steps per trial. Under this step limit, small mini-batch sizes (128) and moderate rollout horizons (16,384) achieved rapid convergence. 

However, during long-term training runs (extending to 1.5 billion steps), smaller mini-batch configurations proved highly unstable. The gradient updates, calculated over a narrow slice of experiences, introduced high variance. Over hundreds of millions of steps, this variance led to gradual policy degradation or catastrophic collapses where the robot would permanently lose its walking gait.

To enforce training stability, the rollout step horizon was scaled to $2^{16} = 65,536$ and the mini-batch size to $2^{12} = 4,096$. Combined with 8 parallel environments, this provides a massive rollout buffer of $524,288$ steps per policy update. Computing the policy gradients over this enormous, highly diverse experience buffer ensures that the gradient steps are low-variance and represent a broad sample of states. This stabilizes the policy updates and guarantees monotonic gait improvement over the long-term training cycle.
