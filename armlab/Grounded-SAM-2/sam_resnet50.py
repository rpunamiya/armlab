import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import supervision as sv
import timm
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim

"""
Hyper parameters
"""
TEXT_PROMPT = "keys."
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_tiny.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATERIAL_THRESHOLD = 0.5

# Material categories (from MINC dataset)
MATERIAL_CLASSES = [
    "fabric", "foliage", "glass", "leather", "metal", "mirror", "paper",
    "plastic", "polished stone", "stone", "wood", "water", "brick", "ceramic",
    "concrete", "food", "frozen", "fur", "hair", "ice", "painted", "sand", "skin"
]

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Load Pretrained MINC Model
model = timm.create_model("resnet50", pretrained=False, num_classes=len(MATERIAL_CLASSES))
checkpoint = torch.load("minc_resnet50.pth", map_location="cpu")  # Load MINC weights
model.load_state_dict(checkpoint)
model.eval()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

text = TEXT_PROMPT

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Set image for SAM2 predictor
        image = torch.tensor(color_image).permute(2, 0, 1).float() / 255.0
        sam2_predictor.set_image(color_image)

        # Detect object with Grounding DINO
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        # Ensure valid boxes exist
        if boxes is None or len(boxes) == 0:
            print("No valid detections found by Grounding DINO.")
            continue

        # Convert bounding boxes to the correct format
        h, w, _ = color_image.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        input_boxes = boxes.cpu().numpy().astype(np.float32)

        if input_boxes.shape[0] == 0:
            print("No valid boxes for SAM2, skipping...")
            continue

        # Get segmentation mask from SAM2
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Ensure mask is valid
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask = masks[0].astype(np.uint8)  # Take the first detected object

        # **Extract Masked Region**
        masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)

        # **Preprocess for Material Classification**
        masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        processed_image = transform(masked_rgb).unsqueeze(0)  # Add batch dimension

        # **Predict Material**
        with torch.no_grad():
            material_logits = model(processed_image)
            material_probs = torch.softmax(material_logits, dim=1)  # Get probabilities
            material_index = torch.argmax(material_logits, dim=1).item()
            predicted_material = MATERIAL_CLASSES[material_index]

            possible_materials = [MATERIAL_CLASSES[i] for i in range(len(MATERIAL_CLASSES)) if material_probs[0, i] > MATERIAL_THRESHOLD]
            print(f"Possible materials: {possible_materials}")

        # **Annotate Image with Material Prediction**
        img = color_image
        detections = sv.Detections(
            xyxy=input_boxes,  # Bounding boxes
            mask=masks.astype(bool),  # Masks
            class_id=np.array(list(range(len(labels))))
        )

        # Overlay material text
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        cv2.putText(
            annotated_frame, f"Material: {predicted_material}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )

        # Show images
        cv2.imshow("Segmented Phone - Material Prediction", annotated_frame)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
