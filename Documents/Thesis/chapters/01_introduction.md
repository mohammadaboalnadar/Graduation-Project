# Chapter 1: Introduction

## 1.1 Background and Motivation
Legged robots are capable of traversing uneven, cluttered, and unstructured terrain that is largely inaccessible to conventional wheeled or tracked platforms, making them well suited for applications such as inspection, search-and-rescue operations, and exploration of environments that are inaccessible to conventional mobile robots. Among legged designs, quadrupedal robots offer a practical balance between mechanical complexity and locomotion stability, and have consequently become a common platform for both industrial deployment and academic research.

Historically, quadrupedal locomotion has been addressed using model-based control techniques, including Model Predictive Control (MPC) and Central Pattern Generators (CPGs). While effective in structured settings, these approaches depend on precise dynamic models, accurate state estimation, and extensive manual tuning, which limits their adaptability when environmental conditions deviate from their design assumptions [1], [2].

Deep Reinforcement Learning (DRL) has emerged as an alternative paradigm that learns locomotion policies directly through trial-and-error interaction, without requiring an explicit analytical model of the robot's dynamics. By formulating the locomotion task as a Markov Decision Process (MDP), a DRL agent learns to map sensory observations to low-level actuation commands so as to maximize a cumulative reward signal [3]. Relative to model-based control, DRL-based methods have demonstrated superior adaptability in complex and previously unseen environments [3], [4].

This thesis presents the development of a model-free locomotion controller for a quadrupedal robot. Using a simulation-only approach, we investigate the training of a Multilayer Perceptron (MLP) network from scratch to achieve stable, coordinated walking on the Unitree A1 quadruped. The findings of this work provide insight into the learning dynamics, reward formulation, and training schedules required to achieve stable legged locomotion under model-free reinforcement learning.

## 1.2 Problem Statement
Deep Reinforcement Learning (DRL) has proven highly effective for synthesizing quadrupedal locomotion policies capable of adapting to varied terrains, largely because it removes the need for an explicit analytical model of the robot's dynamics [3], [4]. However, the effectiveness of a DRL-based controller is determined almost entirely by the design of its reward function. Standard velocity tracking objectives, if implemented in isolation, often prompt the agent to adopt unstable or structurally damaging gaits (such as high-impact stamping or hopping) to maximize speed at the expense of stability.

Optimizing locomotion performance therefore requires balancing tracking commands with stability constraints. In a model-free setting, this balance is difficult to achieve. If stability penalties are too soft, the robot develops awkward, high-impact contact patterns or walks on its knees rather than its feet. If penalties are too severe, the policy falls into the "lazy agent trap," where the robot learns to stand completely still to avoid movement penalties. This highlights the need for structured reward shaping combined with a curriculum training schedule to guide the policy from scratch toward stable, natural walking gaits.

This thesis addresses these challenges by developing and evaluating a PPO-trained MLP policy for the Unitree A1 quadruped within a MuJoCo simulation environment. We focus on formulating a multi-scale velocity tracking reward and a modular curriculum learning callback to enforce stability, orientation, and leg coordination constraints, training a stable walking controller from scratch on a flat surface.

## 1.3 Objectives
The primary objectives of this graduation project are:
1. To develop a Proximal Policy Optimization (PPO)-based locomotion framework for the Unitree A1 quadruped within the MuJoCo simulation environment, enabling the robot to learn walking gaits from scratch.
2. To design a multi-scale Gaussian reward function for velocity tracking combined with gait-shaping penalties (vertical velocity, orientation roll/pitch, and diagonal leg symmetry) to encourage stable trot-like behavior.
3. To implement a Modular Curriculum Learning schedule that gradually introduces joint pose and stability constraints, preventing the policy from falling into local optima during early exploration.
4. To evaluate the tracking accuracy, gait smoothness, and stability of the trained policy on a flat surface in simulation.

## 1.4 Scope and Limitations
This thesis focuses on a simulation-based investigation of quadrupedal locomotion and does not involve any physical robot, hardware implementation, or real-world deployment. All development, training, and evaluation are conducted within the MuJoCo physics engine using the Unitree A1 quadruped model, and no sim-to-real transfer is attempted as part of this work.

The scope of the project covers three stages of development. First, the setup of the Unitree A1 model and environment wrapper within gymnasium. Second, the training of an MLP control policy using PPO with multi-scale rewards and curriculum penalties. Third, the evaluation of the resulting policy's locomotion performance on a flat surface, focusing on tracking error and base stability.

Several aspects fall outside the scope of this thesis. The project does not address perception-based locomotion, such as policies that rely on visual or depth information, and considers only proprioceptive observations. It does not investigate advanced agility behaviors such as parkour, jumping over gaps, or climbing obstacles, nor does it explore hierarchical reinforcement learning, teacher-student distillation, or motion imitation techniques such as Adversarial Motion Priors. The project is also restricted to the Unitree A1 platform and does not generalize its findings to other quadrupedal robots.

A key limitation of this work is that all results are obtained exclusively in simulation, and the extent to which the trained policies would transfer to physical hardware is not evaluated. Therefore, the findings should be interpreted within the assumptions and constraints of the adopted simulation environment.

## 1.5 Thesis Organization
The remainder of this thesis is organized as follows. 
- **Chapter 2: Background and Literature Review** introduces the theoretical concepts underlying reinforcement learning-based quadrupedal locomotion, details PPO and robot kinematics, and reviews related work, concluding with the research gap addressed by this thesis.
- **Chapter 3: System Architecture & Simulation Interface** describes the overall system architecture, the simulation environment setup, the Unitree A1 kinematic profile, and the mathematical specifications of the observation and action spaces.
- **Chapter 4: Reward Engineering & Learning Mechanics** explains the multi-scale Gaussian reward function for velocity tracking, details the mathematical formulation of stability and cosmetic penalties, and analyzes the curriculum design.
- **Chapter 5: Experimental Evaluation & Results** presents the experimental setup, training convergence curves, and locomotion performance metrics of the trained walking policy on a flat surface.
- **Chapter 6: The Sim-to-Real Chasm (Discussion)** identifies unmodeled simulation dynamics, details system identification methodologies, and proposes domain randomization parameters needed for future hardware deployment.
- **Chapter 7: Conclusions & Future Work** summarizes the main findings of the thesis, states the key contributions of the research, and suggests directions for future work.
- **Appendix A: Training Configuration and Hyperparameters** compiles the hyperparameter comparison table (default vs. Optuna vs. final selected parameters) and details the scaling rationale.
- **Appendix B: Key Code Implementations** contains Python code blocks for the custom observation vector formatting and modular curriculum scheduling, alongside links to the public GitHub repository.
- **Appendix C: Supplementary Plots** collects the training convergence logs and evaluation performance charts (velocity tracking and base stability).
