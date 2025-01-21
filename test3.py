import torch
import torchvision.transforms as transforms
import timm
import pyrealsense2 as rs
import cv2
import numpy as np

# Material categories (from MINC dataset)
MATERIAL_CLASSES = [
    "fabric", "foliage", "glass", "leather", "metal", "mirror", "paper",
    "plastic", "polished stone", "stone", "wood", "water", "brick", "ceramic",
    "concrete", "food", "frozen", "fur", "hair", "ice", "painted", "sand", "skin"
]

# Load Pretrained MINC Model
model = timm.create_model("resnet50", pretrained=True, num_classes=len(MATERIAL_CLASSES))
model.eval()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Capture frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess image
        input_tensor = transform(color_image).unsqueeze(0)  # Add batch dimension

        # Predict material class
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Display result
        label = MATERIAL_CLASSES[predicted_class]
        cv2.putText(color_image, f"Material: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Material Classification", color_image)

        # Exit loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
