import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F

"""
Hyper parameters
"""
TEXT_PROMPT = "phone."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Material categories (from MINC dataset)
MATERIAL_CLASSES = [
    "fabric", "foliage", "glass", "leather", "metal", "mirror", "paper",
    "plastic", "polished stone", "stone", "wood", "water", "brick", "ceramic",
    "concrete", "food", "frozen", "fur", "hair", "ice", "painted", "sand", "skin"
]

# Load DINOv2 Model from Meta AI's repo
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE)
dinov2_model.eval()

# Image Preprocessing for DINOv2
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess image for DINOv2
        input_tensor = transform(color_image).unsqueeze(0).to(DEVICE)

        # Get feature embeddings from DINOv2
        with torch.no_grad():
            features = dinov2_model(input_tensor)  # Extract features
            features = F.normalize(features, dim=1)  # Normalize

            # Simulate classification (Replace with trained material classifier)
            material_probs = torch.softmax(features @ torch.randn((features.shape[1], len(MATERIAL_CLASSES)), device=DEVICE), dim=1)
            predicted_label = MATERIAL_CLASSES[torch.argmax(material_probs).item()]

        # Display result
        print(f"Predicted Material: {predicted_label}")
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
