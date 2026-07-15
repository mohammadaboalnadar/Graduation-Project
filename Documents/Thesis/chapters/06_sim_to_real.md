# Chapter 6: The Sim-to-Real Chasm (Discussion)

## 6.1 Unmodeled Simulator Dynamics
While rigid-body simulators like MuJoCo provide a high-fidelity environment for training reinforcement learning policies, a significant gap exists between simulated physics and physical reality, commonly referred to as the **sim-to-real chasm**. A policy trained exclusively in simulation often fails when deployed directly on physical hardware due to unmodeled simulator dynamics:
1. **Motor Latency**: In simulation, joint position commands are executed almost instantaneously or modeled with ideal first-order delays. Real actuators suffer from communication latency (from the main board to the motor driver) and electromagnetic rise times, introducing a lag of $5\text{ ms}$ to $15\text{ ms}$ that can destabilize high-frequency feedback loops.
2. **Gear Backlash**: Quadruped joints use planetary or harmonic gear reducers. These mechanisms possess small clearances between mating gear teeth (backlash), which introduces dead-bands and non-linearities in joint position tracking, particularly during leg direction reversal.
3. **Friction Hysteresis**: Joint bearings and seals introduce dry static friction (Coulomb friction) and velocity-dependent viscous friction. Real systems exhibit hysteresis and temperature-dependent variations that are simplified into constant damping and friction coefficients in simulation.
4. **Battery Voltage Sag**: The torque output of the motors is bounded by the electrical bus voltage. Under high load (e.g., recovering from a fall or jumping), the battery voltage sag decreases the maximum available torque, causing joint torque saturation that the simulated agent does not anticipate.

---

## 6.2 System Identification Methodologies
To minimize the gap between simulation and reality, **System Identification (SysID)** must be performed on the physical Unitree A1 platform to match simulator parameters to physical measurements:
1. **Inertial Parameter Calibration**: The mass, center of mass (CoM), and inertia tensor of the robot parts can be measured using trifilar pendulum setups or estimated using CAD models. The base mass and CoM can also be refined by logging static load cells while the robot is suspended.
2. **Actuator Characterization**: Dynamometer testing can map torque output as a function of joint position, velocity, and bus voltage. This calibration determines the true proportional gain ($K_p$) and damping ($K_d$) coefficients for the joint PD controller, matching the actuator model in Eq. 3.5.
3. **Link Hysteresis Modeling**: By driving the joints through sinusoidal trajectory patterns and measuring the lag between command and feedback, we can estimate joint damping, dry friction, and internal actuator latency.

---

## 6.3 Domain Randomization Framework
Even with rigorous system identification, minor physical mismatches will persist. To ensure the policy is robust enough to transfer to physical hardware, we deploy a **Domain Randomization** framework during training. Instead of training the agent on a single, deterministic physics model, we randomize key physical parameters at the start of each episode, forcing the neural network to learn a generalized policy:

1. **Mass Randomization**: The mass of the trunk ($m_{trunk}$) is randomized by adding a payload offset:
   $$m'_{trunk} = m_{trunk} + \delta_m, \quad \delta_m \sim \mathcal{U}(-1.0, 1.5)\text{ kg} \quad (6.1)$$
   This variation forces the policy to adapt to payload variations (such as onboard sensors or batteries).
2. **Ground Friction Randomization**: The friction coefficient ($\mu$) of the ground plane is randomized:
   $$\mu' \sim \mathcal{U}(0.2, 1.2) \quad (6.2)$$
   This variation exposes the agent to slippery surfaces (such as wet grass or ice) and high-traction surfaces (such as carpet or rubber).
3. **Actuator Gain and Latency Randomization**: The motor proportional gain ($K_p$) and derivative gain ($K_d$) are perturbed:
   $$K'_p \sim \mathcal{U}(90, 110)\text{ Nm/rad}, \quad K'_d \sim \mathcal{U}(1.5, 2.5)\text{ Nms/rad} \quad (6.3)$$
   We also introduce a variable delay buffer (randomized between $0$ and $3$ simulation timesteps) to replicate communication latency.
4. **Sensor Noise Injection**: White noise is added to the observations (joint positions, base roll/pitch, and velocities):
   $$o'_{t} = o_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2) \quad (6.4)$$
   This noise replicates sensor drift, calibration errors, and IMU vibrations, preventing the policy from overfitting to clean simulated signals.

By forcing the PPO policy to find a single set of network weights that achieves stable locomotion across this randomized environment distribution, the resulting policy becomes highly robust and capable of immediate sim-to-real transfer without retraining.
