# Chapter 7: Conclusions & Future Work

## 7.1 Project Summary
This graduation project presents a simulation-based investigation into synthesizing a robust quadrupedal locomotion policy for the Unitree A1 robot using Deep Reinforcement Learning. Operating in a model-free continuous control setting, we successfully trained a Multilayer Perceptron (MLP) control policy from scratch to achieve stable walking on a flat surface. 

The project was executed across three main phases:
1. **Phase I: Environment & Baseline Setup**: We set up the Unitree A1 model within the MuJoCo physics engine and established a gymnasium environment interface with a $50\text{ Hz}$ control loop and $500\text{ Hz}$ physics integration.
2. **Phase II: Gait Shaping and Stability Integration**: We integrated structural stability penalties (base vertical velocity, orientation, diagonal leg symmetry, hip pose similarity, and action acceleration). A modular curriculum learning schedule was deployed to fade in these penalties over the course of training, preventing the policy from falling into the "lazy agent trap" of standing still.
3. **Phase III: Locomotion Evaluation**: We evaluated the trained policy on a flat surface. The results demonstrate that the agent successfully learns a stable, coordinated trot gait that tracks velocity reference commands with low error while maintaining a level base orientation.

The project demonstrates that standard reinforcement learning algorithms (like PPO) can successfully learn complex legged locomotion skills from scratch without requiring reference trajectories or motion imitation priors, provided that the reward function and curriculum are carefully engineered.

---

## 7.2 Key Contributions
The key technical contributions of this research are:
1. **Multi-Scale Gaussian Reward Formulation**: We demonstrated that combining a broad Gaussian kernel (for exploratory gradient) with a tight Gaussian kernel (for steady-state precision) resolves the trade-off between speed of convergence and tracking accuracy, preventing training plateaus.
2. **Analysis of Vanishing Gradients in Reward Plateaus**: We provided a mathematical validation showing why step-like tolerance bands cause vanishing policy gradients, justifying the choice of continuous, smooth Gaussian formulations.
3. **Curriculum Design for Legged Control**: We designed a modular curriculum callback that schedules the activation of stability and leg coordination penalties. This curriculum prevents the policy from falling into local optima during early exploration, enabling stable trot gait emergence.
4. **Baseline for Model-Free Control from Scratch**: We established a clean, model-free baseline for Unitree A1 locomotion using a 50-dimensional proprioceptive observation space and standard MLP policy network without hierarchical planners.

---

## 7.3 Future Recommendations
We recommend several directions for future research:
1. **Terrain Complexity**: The current policy is trained and evaluated on a flat surface. Future work should introduce uneven terrain, stairs, and slopes, requiring the policy to adapt its step height dynamically.
2. **Domain Randomization for Sim-to-Real Transfer**: To deploy this policy on a physical Unitree A1 robot, domain randomization (mass, friction, motor gains, latency, and sensor noise) should be integrated during training to bridge the sim-to-real chasm.
3. **3D Visual Perception**: Currently, the agent relies solely on proprioceptive feedback. Integrating exteroceptive sensors (such as depth cameras or LiDAR scans) would allow the policy to predict obstacles and navigate complex environments.
4. **Energy Efficiency Minimization**: Future iterations could integrate explicit electrical or mechanical power cost penalties (such as torque-squared minimization) to reduce actuator thermal loss and extend battery life.
