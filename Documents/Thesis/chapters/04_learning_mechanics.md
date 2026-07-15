# Chapter 4: Reward Engineering & Learning Mechanics

## 4.1 Phase I: Targeted Velocity Tracking
In reinforcement learning-based locomotion, the primary objective is to train the robot to track reference velocity commands. The design of the velocity tracking reward is critical. If it is too narrow, the robot will struggle to discover a gradient; if too wide, it will track commands inaccurately. 

To resolve this trade-off, this project implements a **multi-scale Gaussian reward function** for velocity tracking:

$$R_{velocity} = w_{broad} \cdot e^{-\alpha_{broad} (v - v_{ref})^2} + w_{tight} \cdot e^{-\alpha_{tight} (v - v_{ref})^2} \quad (4.1)$$

where:
- $v$ is the robot's actual base velocity.
- $v_{ref}$ is the commanded target velocity.
- $\alpha_{broad}$ is a small coefficient (e.g., $0.1$), establishing a broad Gaussian kernel.
- $\alpha_{tight}$ is a large coefficient (e.g., $4.0$), establishing a narrow, tight Gaussian kernel.
- $w_{broad}$ and $w_{tight}$ are corresponding weights.

### Mathematical Rationale
The multi-scale formulation addresses two conflicting demands during training:
1. **Exploration (Broad Gaussian)**: Early in the training process, the policy outputs random actions, and the robot's actual velocity is far from the target ($v_{ref}$). If only a tight Gaussian ($\alpha_{tight} = 4.0$) were used, the reward value for these exploratory steps would be near zero, and its gradient would be negligible:
   $$\lim_{|v - v_{ref}| \to \infty} \frac{\partial R_{velocity}}{\partial v} \approx 0 \quad (4.2)$$
   A broad Gaussian ($\alpha_{broad} = 0.1$) ensures that even for poor tracking performance, the agent receives a non-zero reward with a usable gradient, guiding the policy towards the target velocity.
2. **Exploitation & Precision (Tight Gaussian)**: Once the agent learns to walk near the target velocity, the gradient of the broad Gaussian becomes flat near the peak ($v \approx v_{ref}$). The tight Gaussian ($\alpha_{tight} = 4.0$) provides a steep gradient near the target, penalizing small deviations and forcing the policy to learn high-precision tracking.

---

## 4.2 Pitfalls of Reward Plateaus (The Flat Tolerance Zone)
A common mistake in reward design is using step functions or "if-else" tolerance bands to reward tracking. For example, a designer might define a reward function as:

$$R_{step}(e) = \begin{cases} 1.0 & \text{if } |e| \le \delta \\ 0.0 & \text{if } |e| > \delta \end{cases} \quad (4.3)$$

where $e = v - v_{ref}$ represents the tracking error, and $\delta$ is a predefined tolerance threshold.

### Mathematical Proof of Vanishing Gradients
Policy gradient algorithms (such as PPO) optimize the policy parameters $\theta$ by maximizing the objective:

$$\nabla_\theta J(\theta) = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right] \quad (4.4)$$

where the advantage estimate $\hat{A}_t$ is directly computed from the discounted sum of rewards. 

For the step reward $R_{step}(e)$, the derivative of the reward with respect to the tracking error is:

$$\frac{\partial R_{step}(e)}{\partial e} = \begin{cases} 0 & \text{if } |e| \ne \delta \\ \text{undefined} & \text{if } |e| = \delta \end{cases} \quad (4.5)$$

Because the derivative is zero almost everywhere, the gradient of the reward vanishes:
1. **Outside the tolerance zone ($|e| > \delta$)**: The reward is a flat plateau at $0.0$. The agent receives no feedback indicating whether an action decreased the error (e.g., reducing $e$ from $2.0 \rightarrow 1.0$), stalling training.
2. **Inside the tolerance zone ($|e| \le \delta$)**: The reward is a flat plateau at $1.0$. The agent receives no feedback indicating whether an action improved tracking precision (e.g., reducing $e$ from $0.05 \rightarrow 0.01$). Consequently, the agent settles on the sloppy upper bound of the tolerance limit ($|e| \approx \delta$).

In contrast, the Gaussian formulation $R(e) = e^{-\alpha e^2}$ provides a continuous, non-zero gradient everywhere:

$$\frac{\partial R(e)}{\partial e} = -2\alpha e \cdot e^{-\alpha e^2} \quad (4.6)$$

This gradient vanishes only when $e = 0$, guaranteeing that the optimization process receives smooth guidance throughout the entire state space.

---

## 4.3 Gait Shaping & Stability Penalties
While velocity tracking is the primary objective, training a policy purely on velocity tracking results in unnatural, structurally destructive gaits (such as hopping or knee-walking). To shape the gait and enforce coordination, we incorporate several soft-constraint penalties directly into the reward scalarization:

### 4.3.1 Hip Pose Similarity Penalty
To prevent the robot from walking on its knees or adopting excessively wide or awkward stances, we penalize deviations of the joint positions from the default standing pose:

$$P_{pose} = - \sum_{i \in \text{joints}} (q_i - q_{default, i})^2 \quad (4.7)$$

Specifically, we apply a hip-pose-only similarity penalty to constrain the abduction joints:

$$P_{hip\_pose} = - \sum_{j \in \text{hips}} (q_j - q_{default, j})^2 \quad (4.8)$$

where hips correspond to the four abduction joints. This forces the legs to swing primarily in the sagittal plane, preventing self-collisions and maintaining structural alignment.

### 4.3.2 Base Vertical Velocity Penalty
To prevent the agent from jumping or hopping forward, we penalize the vertical velocity of the base:

$$P_{vert\_vel} = - \dot{z}^2_{base} \quad (4.9)$$

where $\dot{z}_{base}$ is the linear velocity of the trunk along the vertical axis. This restricts vertical oscillation, bounding the torso movement to a flat horizontal plane and generating a smoother walking gait.

### 4.3.3 Base Orientation Penalties
To keep the robot upright and prevent rollovers, we penalize the rolling pitch and roll of the base trunk. Crucially, the penalty is computed using the exponential moving average (EMA) of these angles (the orientation skew variables $\bar{\theta}_{pitch}$ and $\bar{\theta}_{roll}$ defined in Eq. 3.3) rather than their instantaneous, single-step values:

$$P_{orientation} = - (\bar{\theta}_{pitch}^2 + \bar{\theta}_{roll}^2) \quad (4.10)$$

where $\bar{\theta}_{pitch}$ and $\bar{\theta}_{roll}$ are the running orientation skews. Bounding the cumulative orientation skew stabilizes the torso, which is critical for carrying sensor payloads. As noted in Section 3.3, because this penalty depends on the historical accumulation of pitch and roll, passing the running skew values back to the agent as observations is mathematically necessary. It ensures that the state space remains Markovian, preventing the policy from experiencing partial observability (POMDP) regarding its own cumulative penalty metrics.

### 4.3.4 Diagonal Leg Symmetry Penalty
To encourage the discovery of a natural walking trot, we enforce symmetry between diagonal pairs of legs:

$$P_{symmetry} = - \left( \| q_{FR} - \bar{q}_{RL} \|^2 + \| q_{FL} - \bar{q}_{RR} \|^2 \right) \quad (4.11)$$

where $q_{leg}$ represents the 3 joint positions of the corresponding leg, and $\bar{q}_{leg}$ represents the joint positions with the hip abduction coordinate negated for mirroring. This penalty coordinates diagonal support pairs (Front Right & Rear Left, Front Left & Rear Right), which is the physical basis of a trot gait.

### 4.3.5 Action Acceleration Penalty
To prevent high-frequency oscillations in actuator commands (motor jitter), we penalize the second derivative of the action vector:

$$P_{accel} = - \sum_{k=1}^{12} (a_{k, t} - 2a_{k, t-1} + a_{k, t-2})^2 \quad (4.12)$$

where $a_{k,t}$ is the action command output for motor $k$ at time step $t$. Minimizing this term forces PPO to select smooth joint trajectories, reducing physical motor wear and actuator stress.

---

## 4.4 Optimization Scaling & Avoidance of Policy Devolution
Integrating multiple stability penalties introduces the risk of **policy devolution**, commonly referred to as the **"lazy agent trap."**

### The Lazy Agent Trap
If the penalty weights are fully active from the beginning of training, the policy will discover a local optimum: **refusing to move at all**. By remaining standing still or collapsing, the agent incurs zero vertical velocity, orientation, and symmetry penalties:

$$P_{vert\_vel} \approx 0, \quad P_{orientation} \approx 0, \quad P_{symmetry} \approx 0 \quad (4.13)$$

While this behavior fails the tracking task, it avoids the large movement penalties that an exploratory walking agent would incur. Because PPO seeks to maximize cumulative rewards, this "lazy" strategy dominates early training before a viable gait is discovered, preventing the robot from learning to walk.

### Curriculum Learning and Penalty Fades
To prevent policy devolution, this project implements a **Modular Curriculum Learning** schedule. The penalties are not active at the start of training. Instead, we use linear penalty fades, where the weight of each penalty is multiplied by a fade factor $f_{penalty} \in [0.0, 1.0]$ that scales with the global training steps:

$$R_{total} = R_{task} + \sum_{k} f_k(t) \cdot w_k \cdot P_k \quad (4.14)$$

The curriculum parameters configured in `lab/train.py` are:
- **Vertical Velocity ($v_z$) Penalty**: Fades in from $50\text{M}$ to $200\text{M}$ steps.
- **Orientation (Roll & Pitch) Penalty**: Fades in from $200\text{M}$ to $400\text{M}$ steps.
- **Diagonal Symmetry Penalty**: Fades in from $300\text{M}$ to $500\text{M}$ steps.
- **Hip Joint Pose Similarity**: Fades in from $720\text{M}$ to $850\text{M}$ steps.

By gradually fading in these penalties, the PPO agent first discovers a fast forward walking gait and then refines it into a symmetric, stable trot without falling into local optima.
