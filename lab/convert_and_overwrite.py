import os
from PIL import Image

mapping = {
    "a1_kinematics.png": r"C:\Users\Airte\.gemini\antigravity\brain\7f5bb913-2d98-40db-a123-7cb30a867429\a1_kinematics_1784237821458.jpg",
    "control_architecture.png": r"C:\Users\Airte\.gemini\antigravity\brain\7f5bb913-2d98-40db-a123-7cb30a867429\control_architecture_1784237831571.jpg",
    "gaussian_rewards.png": r"C:\Users\Airte\.gemini\antigravity\brain\7f5bb913-2d98-40db-a123-7cb30a867429\gaussian_rewards_1784237839776.jpg",
    "curriculum_timeline.png": r"C:\Users\Airte\.gemini\antigravity\brain\7f5bb913-2d98-40db-a123-7cb30a867429\curriculum_timeline_1784237849066.jpg"
}

dest_dir = r"D:\Files\Scripts\py\Graduation Project\Figures"

for filename, src_path in mapping.items():
    dest_path = os.path.join(dest_dir, filename)
    try:
        if os.path.exists(src_path):
            with Image.open(src_path) as img:
                img.save(dest_path, "PNG")
            print(f"Successfully converted and copied to {dest_path}")
        else:
            print(f"Source file not found: {src_path}")
    except Exception as e:
        print(f"Error copying {filename}: {e}")
