import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import io

# SB3/Pytorch compatibility stream fix
_original_load = torch.load
def _safe_load(f, *args, **kwargs):
    if hasattr(f, "read"):
        buffer = io.BytesIO(f.read())
        return _original_load(buffer, *args, **kwargs)
    return _original_load(f, *args, **kwargs)
torch.load = _safe_load

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import UnitreeA1Env

def evaluate_and_plot():
    # Use the 720M checkpoint (720,000,000 steps)
    checkpoint_path = r".\Models\checkpoints\v19.5_2\720000000_steps.zip"
    vecnorm_path = r".\Models\checkpoints\v19.5_2\720000000_steps_vecnorm.pkl"
    xml_path = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
    
    figures_dir = r".\Figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create the environment with render_mode=None for headless evaluation
    raw_env = UnitreeA1Env(xml_path, render_mode=None, max_episode_steps=500)
    venv = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(vecnorm_path, venv)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(checkpoint_path, env=env)
    
    obs = env.reset()
    
    # Set the tracking commands (target vx = 5.0 m/s, target height = 0.28m)
    # Target vx = 5 * ref_vel[0] -> 5 * 1.0 = 5.0 m/s
    raw_env.set_commands(vx=1.0, vy=0.0, wz=0.0, height=0.0)
    
    time_steps = []
    ref_vxs = []
    act_vxs = []
    ref_vys = []
    act_vys = []
    pitches = []
    rolls = []
    heights = []
    torques = []
    cot_values = []
    
    dt = raw_env.dt
    gravity = 9.81
    robot_mass = 12.453  # calculated total mass of A1 model
    
    print("Running simulation...")
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract metrics
        world_vel = raw_env.data.qvel[0:2]
        yaw, pitch, roll = raw_env._get_euler()
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        actual_vel_x = cos_yaw * world_vel[0] - sin_yaw * world_vel[1]
        actual_vel_y = sin_yaw * world_vel[0] + cos_yaw * world_vel[1]
        
        ref_vel_x = 5.0 * raw_env.ref_vel[0]
        ref_vel_y = 2.5 * raw_env.ref_vel[1]
        
        z_height = raw_env.data.qpos[2]
        
        # Calculate joint torques (PD law)
        # action is mapped to target joint position
        mapped_action = raw_env.default_dof_pos + (action[0] * raw_env.action_scale)
        q = raw_env.data.qpos[7:19]
        qvel = raw_env.data.qvel[6:18]
        
        Kp = 100.0
        Kd = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0])  # K_d gains
        joint_torques = Kp * (mapped_action - q) - Kd * qvel
        torque_norm = np.linalg.norm(joint_torques)
        
        # Calculate Cost of Transport (CoT)
        # Power P = sum(|tau_i * qvel_i|)
        mech_power = np.sum(np.abs(joint_torques * qvel))
        actual_speed = np.sqrt(actual_vel_x**2 + actual_vel_y**2)
        if actual_speed > 1e-3:
            cot = mech_power / (robot_mass * gravity * actual_speed)
        else:
            cot = 0.0
            
        time_steps.append(step * dt)
        ref_vxs.append(ref_vel_x)
        act_vxs.append(actual_vel_x)
        ref_vys.append(ref_vel_y)
        act_vys.append(actual_vel_y)
        pitches.append(np.degrees(pitch))
        rolls.append(np.degrees(roll))
        heights.append(z_height)
        torques.append(torque_norm)
        cot_values.append(cot)
        
    print("Simulation finished. Calculating statistics...")
    rmse_vx = np.sqrt(np.mean((np.array(act_vxs) - np.array(ref_vxs))**2))
    rmse_vy = np.sqrt(np.mean((np.array(act_vys) - np.array(ref_vys))**2))
    var_pitch = np.var(pitches)
    var_roll = np.var(rolls)
    mean_cot = np.mean([c for c in cot_values if c > 0 and c < 15.0]) # filter outliers
    mean_torque = np.mean(torques)
    
    print(f"Forward Velocity RMSE: {rmse_vx:.4f} m/s")
    print(f"Lateral Velocity RMSE: {rmse_vy:.4f} m/s")
    print(f"Torso Pitch Variance: {var_pitch:.4f} deg^2")
    print(f"Torso Roll Variance: {var_roll:.4f} deg^2")
    print(f"Mean Cost of Transport: {mean_cot:.4f}")
    print(f"Mean Torque Norm: {mean_torque:.4f} Nm")
    
    # Save statistics to a text file for thesis results
    with open(os.path.join(figures_dir, "evaluation_stats.txt"), "w") as f:
        f.write(f"Evaluation of Checkpoint: {checkpoint_path}\n")
        f.write(f"Forward Velocity Tracking RMSE: {rmse_vx:.6f} m/s\n")
        f.write(f"Lateral Velocity Tracking RMSE: {rmse_vy:.6f} m/s\n")
        f.write(f"Torso Pitch Variance: {var_pitch:.6f} deg^2\n")
        f.write(f"Torso Roll Variance: {var_roll:.6f} deg^2\n")
        f.write(f"Mean Cost of Transport (CoT): {mean_cot:.6f}\n")
        f.write(f"Mean Joint Torque Norm: {mean_torque:.6f} Nm\n")
        
    # Plot 1: Velocity Tracking
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, ref_vxs, '--', color='green', label='Target Forward Velocity (vx)')
    plt.plot(time_steps, act_vxs, '-', color='#E8593C', label='Actual Forward Velocity (vx)')
    plt.plot(time_steps, ref_vys, '--', color='blue', label='Target Lateral Velocity (vy)')
    plt.plot(time_steps, act_vys, '-', color='#5B6AD0', label='Actual Lateral Velocity (vy)')
    plt.title("Quadrupedal Locomotion: Velocity Tracking Performance (720M Checkpoint)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "evaluation_velocity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Base Stability
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(time_steps, pitches, color='#E8593C', label='Pitch')
    ax1.plot(time_steps, rolls, color='#5B6AD0', label='Roll')
    ax1.set_ylabel("Orientation (Degrees)")
    ax1.set_title("Base Stability and Attenuation of Torso Rocking")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_steps, heights, color='green', label='Height (z)')
    ax2.axhline(0.28, color='black', linestyle='--', label='Target Height (0.28m)')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Torso Height (meters)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "evaluation_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Plots exported successfully to the Figures directory.")

if __name__ == "__main__":
    evaluate_and_plot()
