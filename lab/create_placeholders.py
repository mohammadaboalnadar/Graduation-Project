import os
from PIL import Image, ImageDraw, ImageFont

def create_placeholder(filename, title):
    figures_dir = r".\Figures"
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    # Create a 800x450 image (16:9) with a light grey background
    img = Image.new('RGB', (800, 450), color='#EFEFEF')
    draw = ImageDraw.Draw(img)
    
    # Draw a thin border
    draw.rectangle([10, 10, 790, 440], outline='#CCCCCC', width=2)
    
    # Write text
    text_placeholder = "PLACEHOLDER"
    text_instruction = "Replace with your final diagram/image"
    
    # Since we might not have a default TTF font path cross-platform,
    # we'll draw lines/simple boxes or use the default bitmap font.
    # To make it look nice, we can draw a decorative box.
    draw.rectangle([50, 50, 750, 400], outline='#BBBBBB', width=1)
    
    # Let's write the text using default font
    try:
        # draw text
        draw.text((400, 150), text_placeholder, fill='#999999', anchor="mm")
        draw.text((400, 220), title, fill='#333333', anchor="mm")
        draw.text((400, 290), text_instruction, fill='#666666', anchor="mm")
    except Exception:
        # Fallback if anchor is not supported on old PIL versions
        draw.text((350, 140), text_placeholder, fill='#999999')
        draw.text((300, 210), title, fill='#333333')
        draw.text((250, 280), text_instruction, fill='#666666')
        
    img.save(filepath)
    print(f"Created placeholder: {filepath}")

if __name__ == "__main__":
    create_placeholder("a1_kinematics.png", "Unitree A1 Kinematic Model & Joint Frames")
    create_placeholder("control_architecture.png", "Closed-Loop PD & RL Control Architecture")
    create_placeholder("gaussian_rewards.png", "Gaussian Reward Scale Functions")
    create_placeholder("curriculum_timeline.png", "Modular Curriculum Timeline Schedules")
