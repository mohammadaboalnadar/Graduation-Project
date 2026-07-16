# Chapter 3: System Architecture & Simulation Interface

## 3.1 The MuJoCo Simulation Environment
Robotic reinforcement learning tasks benefit from simulation platforms that offer high-speed execution alongside physics fidelity. This project utilizes the **MuJoCo (Multi-Joint dynamics with Contact)** physics engine. MuJoCo uses a convex contact solver that resolves multi-body constraints via a mathematical programming approach, rather than the spring-damper contact approximations used in older engines. This formulation guarantees physical plausibility and prevents penetration artifacts in rigid-body contact solver equations.

The simulation and control loop are structured hierarchically:
- **Simulation Timestep ($\Delta t_{sim}$)**: The internal physics simulation runs at a timestep of $\Delta t_{sim} = 0.002\text{ s}$ (500 Hz). This duration is chosen to ensure numerical stability in contact physics and joint acceleration integration, preventing numerical explosions during high-velocity collisions.
- **Control Frequency ($f_{control}$)**: The high-level reinforcement learning agent runs at a control frequency of $f_{control} = 50\text{ Hz}$. To interface these two rates, the environment uses a substep integration factor of $n_{substeps} = 10$. That is, for every action command output by the neural network, MuJoCo steps the physics forward 10 times:
  $$dt = \Delta t_{sim} \times n_{substeps} = 0.002\text{ s} \times 10 = 0.02\text{ s} \quad (3.1)$$
  A 50 Hz control rate is highly practical for legged locomotion. It mirrors the onboard processor bandwidth of physical quadruped computers (such as NVIDIA Jetson platforms) and is fast enough to react to external disturbances without overloading the processor.

---

## 3.2 Kinematic Profile of the Unitree A1
The robot model deployed in the simulation environment is the **Unitree A1**, a commercial-grade quadrupedal platform. The physical parameters of the robot are modeled from the manufacturer's specification files:
- **Total Robot Mass ($m$)**: The total simulated mass of the robot is approximately $12.45\text{ kg}$, which comprises:
  - Trunk (base body): $4.713\text{ kg}$
  - Hips (4 units): $0.696\text{ kg}$ per hip
  - Thighs (4 units): $1.013\text{ kg}$ per thigh
  - Calves (4 units): $0.226\text{ kg}$ per calf
- **Degrees of Freedom**: The robot has 12 active joints. Each of the four legs (Front Right, Front Left, Rear Right, Rear Left) is actuated by 3 motors:
  1. Hip joint (abduction/adduction)
  2. Thigh joint (flexion/extension)
  3. Knee joint (flexion/extension)
- **Actuator Capabilities**: The 12 motors are position-controlled actuators with torque limits of $\pm 33.5\text{ Nm}$ and maximum joint angular velocities around $21\text{ rad/s}$. The joints are constrained to safe physical limits:
  - Hip abduction: $-0.803$ to $+0.803\text{ rad}$
  - Thigh flexion: $-1.047$ to $+4.189\text{ rad}$
  - Knee flexion: $-2.697$ to $-0.916\text{ rad}$

![Unitree A1 Kinematics and coordinate frames.](../Figures/a1_kinematics.png)
*Figure 3.1: Unitree A1 kinematic profile rendered directly in the MuJoCo simulation environment, showing coordinate frames on links, joints, and default standing pose.*
\label{fig:a1_kinematics}

---

## 3.3 Observation Space Specification
The observation space represents the sensory information fed to the actor-critic policy networks at each control step. The policy processes a **50-dimensional observation vector**, which contains proprioceptive measurements, command states, and historical action information. 

Each component is normalized or scaled to stay approximately in the range of $[-1.0, 1.0]$ to facilitate neural network training stability:

1. **Base Linear Velocities (3 dimensions)**: The linear velocity of the robot's center of mass along the $x, y, z$ axes in the robot's local coordinate frame, divided by $3.0\text{ m/s}$ (expected maximum range).
2. **Base Angular Velocities (3 dimensions)**: The rotational velocities (roll, pitch, yaw rates) of the robot base, divided by $2.0\text{ rad/s}$.
3. **Base Orientation Angles (2 dimensions)**: The roll and pitch angles of the robot base relative to the horizontal gravity vector, divided by $\pi$. Using roll and pitch directly avoids Euler angle gimbal lock while ignoring the absolute yaw angle, which is redundant since the robot must navigate relative to a reference coordinate system.
4. **Joint Positions (12 dimensions)**: The current positions of the 12 active joints. Rather than using raw joint angles, these are expressed as deviations from the default standing pose ($q_{default}$) and divided by the joint action scale parameter ($0.8\text{ rad}$):
   $$\bar{q} = \frac{q - q_{default}}{action\_scale} \quad (3.2)$$
   where $q_{default}$ consists of $[-0.1, 0.8, -1.5]\text{ rad}$ for the right legs and $[0.1, 0.8, -1.5]\text{ rad}$ for the left legs.
5. **Joint Velocities (12 dimensions)**: The current velocities of the 12 joints, divided by $21.0\text{ rad/s}$ (the A1's physical motor limit).
6. **Previous Action Commands (12 dimensions)**: The actions generated by the policy at the previous control step (already bounded within $[-1.0, 1.0]$).
7. **Orientation Skew (2 dimensions)**: An exponential moving average (EMA) of the base pitch and roll:
   $$skew_t = (1 - \alpha_{skew}) \cdot skew_{t-1} + \alpha_{skew} \cdot \theta_{roll, pitch} \quad (3.3)$$
   where $\alpha_{skew} = 0.1$. This term is critical for maintaining the Markov property of the system state space. Because the orientation penalties (described in Chapter 4) are computed based on this cumulative running skew rather than immediate step orientation, the environment would technically be partially observable (a POMDP) if the agent only received step-level roll and pitch. Passing the running skew directly to the policy network provides it with explicit knowledge of its stability history. This cumulative representation effectively acts as a low-pass filter: it allows the high-frequency body oscillations (natural rocking) necessary for dynamic weight shifting during trotting to pass through without massive penalties, while enabling the PPO agent to proactively damp out low-frequency, persistent imbalances (unbalanced leaning or drift).
8. **Reference Commands (3 dimensions)**: The commanded forward velocity ($v_x$), lateral velocity ($v_y$), and yaw rate ($\omega_z$), representing the task specification.
9. **Reference Height (1 dimension)**: The commanded body height target ($h_{ref}$).

$$3 + 3 + 2 + 12 + 12 + 12 + 2 + 3 + 1 = 50\text{ dimensions}$$

---

## 3.4 Action Space & Actuator Control
The action space of the policy is a continuous **12-dimensional vector** bounded within $[-1.0, 1.0]$. The neural network does not output raw joint torques directly. Instead, it outputs target joint positions, which are mapped to physical angles and processed by a low-level Proportional-Derivative (PD) controller.

The mapping from the policy action vector $a_t \in [-1, 1]^{12}$ to the target joint positions $q_{target} \in \mathbb{R}^{12}$ is defined as:

$$q_{target} = q_{default} + a_t \cdot action\_scale \quad (3.4)$$

where $action\_scale = 0.8\text{ rad}$. This scaling prevents the policy from commanding angles outside the robot's physical workspace, acting as an implicit safety constraint.

The target joint positions are sent to position actuators modeled in MuJoCo, which simulate a Joint PD controller running at the physics rate ($500\text{ Hz}$). The torque commanded at each actuator is computed as:

$$\tau = K_p (q_{target} - q) - K_d \dot{q} \quad (3.5)$$

where:
- $K_p$ is the proportional gain, set to $100\text{ Nm/rad}$.
- $K_d$ is the derivative damping coefficient, set to $2.0\text{ Nms/rad}$ for hip and knee joints, and $1.0\text{ Nms/rad}$ for hip abduction joints.
- $q$ and $\dot{q}$ represent the current joint position and joint velocity, respectively.

This joint-level PD formulation mirrors the actuator architecture of physical legged robots, where high-frequency motor control loops run independently on dedicated hardware, receiving lower-frequency joint target commands from the primary computer.

![RL Policy and low-level joint PD control loop architecture.](../Figures/control_architecture.png)
*Figure 3.2: High-level control loop architecture, tracing observation mapping, RL policy target generation, and low-level actuator PD torque computation.*
\label{fig:control_loop}

---

## 3.5 Tools and Technologies
The development, training, and evaluation of the quadrupedal locomotion policy are executed using a standardized stack of open-source software libraries, running on consumer-grade workstation hardware.

### 3.5.1 Software Libraries and Frameworks
*   **Programming Language & Runtime**: Python (v3.12.0) is utilized to manage the custom virtual environment (`.venv`) and coordinate the training and evaluation execution scripts.
*   **Simulation Engine (MuJoCo)**: The rigid-body physics simulation is powered by MuJoCo (Multi-Joint dynamics with Contact), which resolves the contact dynamics, joint torques, and kinematics of the Unitree A1 model.
*   **Environment API (Gymnasium)**: The custom robot environment is wrapped using Gymnasium (v0.29), providing standard API endpoints (`step`, `reset`, `observation_space`, and `action_space`) for the RL training loops.
*   **Reinforcement Learning Library (Stable-Baselines3)**: The training pipeline utilizes Stable-Baselines3 (v2.0) to implement the Proximal Policy Optimization (PPO) algorithm. SB3 handles parallelized environment execution via `SubprocVecEnv` and manages online observation and reward scaling via the `VecNormalize` wrapper.
*   **Deep Learning Back-End (PyTorch)**: Neural network compilation and gradient backpropagation are executed using PyTorch (v2.4). The actor and critic are modeled as Multilayer Perceptrons (MLPs) with two hidden layers of 256 units each.
*   **Experiment Tracking and Logging**: Weights & Biases (WandB) and TensorBoard are used for real-time visualization and logging of convergence curves (reward, value loss, approximate KL divergence) and modular curriculum schedules.

### 3.5.2 Computational Workstation Specifications
To ensure the reproducibility of the training process under accessible hardware environments, training was performed locally on a consumer-grade laptop. The workstation hardware details are:
*   **Processor (CPU)**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 physical cores, 16 logical threads, 24.0 MB L3 Cache, operating at boost speeds up to 3.82 GHz). The CPU manages the multi-processing overhead for running the 8 parallel MuJoCo gymnasium environments.
*   **System Memory (RAM)**: 32.0 GB DDR4 Dual-Channel @ 3200 MT/s. The high memory capacity ensures that large rollout buffers ($65,536$ steps $\times$ 8 environments) are held in memory without disk swapping.
*   **Graphics Processor (GPU)**: NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB dedicated VRAM). The GPU accelerates the PyTorch neural network forward and backward training passes during policy gradient updates.

