import os
import numpy as np
import mujoco
from PIL import Image

def render_robot_scene():
    xml_path = r".\external\mujoco_menagerie\unitree_a1\scene.xml"
    figures_dir = r".\Figures"
    
    # Read XML
    with open(xml_path, "r") as f:
        xml_content = f.read()
        
    # Replace skybox texture to be flat white
    xml_content = xml_content.replace(
        '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        '<texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="512" height="512"/>'
    )
    
    # Remove floor geom
    xml_content = xml_content.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" priority="1" friction="10 0.005 0.0001"/>',
        ''
    )
    
    # Write a temporary XML file in the same directory to resolve inclusions
    temp_xml_path = r".\external\mujoco_menagerie\unitree_a1\temp_scene.xml"
    with open(temp_xml_path, "w") as f:
        f.write(xml_content)
        
    try:
        model = mujoco.MjModel.from_xml_path(temp_xml_path)
        data = mujoco.MjData(model)
        
        # Put the robot in default standing pose
        default_dof_pos = np.array([
            -0.1, 0.8, -1.5,  # FR
             0.1, 0.8, -1.5,  # FL
            -0.1, 0.8, -1.5,  # RR
             0.1, 0.8, -1.5   # RL
        ])
        data.qpos[7:19] = default_dof_pos
        
        # Step once to update kinematics
        mujoco.mj_forward(model, data)
        
        # Create Renderer (640x360 to respect default framebuffer limits)
        renderer = mujoco.Renderer(model, height=360, width=640)
        
        # Configure frames and visual scales
        scene_option = mujoco.MjvOption()
        scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY
        
        # Make the axis cylinders much thinner and shorter (default framewidth is 0.1, framelength is 1.0)
        model.vis.scale.framewidth = 0.02   # 5x thinner than default
        model.vis.scale.framelength = 0.3   # 3.3x shorter than default
        
        # Set a very narrow FOV (telephoto/orthographic effect)
        model.vis.global_.fovy = 8.0        # 8 degrees field of view (default is 45)
        
        # Set the camera view to look from the front-left, pushed far back
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 5.5                  # Pushed far back (5.5 meters)
        cam.elevation = -12
        cam.azimuth = 225                   # Front-left view
        cam.lookat = np.array([0.0, 0.0, 0.25])  # Target torso center height
        
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        image_rgb = renderer.render()
        
        dest_path = os.path.join(figures_dir, "a1_kinematics.png")
        img = Image.fromarray(image_rgb)
        img.save(dest_path)
        print("Success rendering scene")
    finally:
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)

if __name__ == "__main__":
    render_robot_scene()
