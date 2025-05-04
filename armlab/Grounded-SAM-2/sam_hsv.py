#!/usr/bin/env python3

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
import tqdm
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

class VisualDetector:
    def __init__(self):
        
        SAM2_MODEL_CONFIG = (
            "/home/rohan/xarm_ros2_ws/src/armlab/"
            "Grounded-SAM-2/sam2/configs/sam2.1/"
            "sam2.1_hiera_t.yaml"
        )
        
        SAM2_CHECKPOINT = (
            "/home/rohan/xarm_ros2_ws/src/armlab/"
            "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"
        )
        SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        GDINO_DEVICE = "cpu"
        
        sam2_model = build_sam2(
            SAM2_MODEL_CONFIG,
            SAM2_CHECKPOINT,
            device=SAM_DEVICE,
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # """
        # Hyper parameters
        # """
        TEXT_PROMPT = "ball."
        # # SAM2_CHECKPOINT = "/home/rohan/xarm_ros2_ws/src/armlab/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt"
        # SAM2_MODEL_CONFIG = "/home/rohan/xarm_ros2_ws/src/armlab/Grounded-SAM-2/sam2/sam2_hiera_t.yaml"
        GROUNDING_DINO_CONFIG = "/home/rohan/xarm_ros2_ws/src/armlab/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = "/home/rohan/xarm_ros2_ws/src/armlab/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        # # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2"))
        # config_name = "sam2_hiera_t.yaml"          # just the file name
        
        # if GlobalHydra.instance().is_initialized():
        #     GlobalHydra.instance().clear()

        # with initialize_config_dir(config_dir=config_dir, job_name="sam2_setup"):
        #     cfg = compose(config_name=config_name)
        
        # sam2_model   = build_sam2(cfg, SAM2_CHECKPOINT, device=DEVICE)
        # sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=GDINO_DEVICE
        )
        grounding_model.to(GDINO_DEVICE)

        # Initialize SAM2 and Grounding DINO models
        self.text_prompt = TEXT_PROMPT
        self.grounding_model = grounding_model
        self.sam_device = SAM_DEVICE
        self.gdino_device = GDINO_DEVICE
        self.box_threshold = BOX_THRESHOLD
        self.material_threshold = 0.5
        self.text_threshold = TEXT_THRESHOLD

    def get_mask(self, image):
        self.sam2_predictor.set_image(image)

        # For SAM2 we’ll use self.sam_device:
        image_tensor_sam = torch.from_numpy(image).permute(2,0,1).float().to(self.sam_device)/255.0

        # Detect with GroundingDINO **on CPU**:
        image_tensor_gd = image_tensor_sam.detach().cpu()
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image_tensor_gd,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.gdino_device,
        )

        # Ensure valid boxes exist
        if boxes is None or len(boxes) == 0:
            print("No valid detections found by Grounding DINO.")
            return np.zeros_like(image)  # Return a blank 3-channel image

        # Convert bounding boxes to the correct format
        h, w, _ = image.shape
        boxes = boxes * torch.tensor([w, h, w, h], device=self.gdino_device)
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        input_boxes = boxes.cpu().numpy().astype(np.float32)

        if input_boxes.shape[0] == 0:
            print("No valid boxes for SAM2, skipping...")
            return np.zeros_like(image)  # Return a blank 3-channel image

        # Get segmentation mask from SAM2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Ensure mask is valid
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        mask = (masks[0] * 255).astype(np.uint8)  # Take the first detected object

        # Convert the mask to a 3-channel format
        mask_3channel = cv2.merge([mask, mask, mask])  # Convert from (H, W) to (H, W, 3)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, mask_3channel)

        if np.count_nonzero(mask) == 0:          # in case SAM2 returned an empty mask
            center_xy = None
        else:
            # Option A – moments (robust, fast)
            M = cv2.moments(mask)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center_xy = (cx, cy)

        # (equivalently, Option B – numpy indices)
        # ys, xs = np.where(mask > 0)
        # cx, cy = int(xs.mean()), int(ys.mean())
        # center_xy = (cx, cy)

        # Write the masked image to a file
        # cv2.imwrite("masked_image.png", masked_image)

        return masked_image, mask, center_xy

    def estimate_light_intensity(self, image_gray):
        """
        Estimate the light intensity by assuming the brightest region is fully illuminated.
        """
        return np.percentile(image_gray, 99.9)  # Take 99.9th percentile as max illumination

    def extract_reflectance(self, masked_image, mask):
        """
        Extracts the relative reflectance of an object from a masked image.
        
        Args:
            masked_image (np.ndarray): A masked image with the object of interest.
        
        Returns:
            float: The average relative reflectance of the object.
        """
        # Convert the image to grayscale (intensity)
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        # Compute the estimated max intensity of the masked region
        est_light_intensity = self.estimate_light_intensity(gray_image)

        # Normalize the grayscale image to [0, 1] range
        normalized_intensity = gray_image / est_light_intensity

        # Compute the average reflectance over the masked region
        refl_mean = np.mean(normalized_intensity[mask > 0])  # Average over non-zero pixels
        refl_var = np.var(normalized_intensity[mask > 0])  # Variance over non-zero pixels

        return refl_mean, refl_var

    def average_hsv_masked(self, masked_image):
        """ Extracts the average HSV values from a masked region of an image. """

        # Convert the masked image from BGR to HSV
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        # Create a binary mask of the foreground (non-black pixels)
        foreground_mask = cv2.inRange(masked_image, (1, 1, 1), (255, 255, 255))  # Mask non-black pixels

        # Use the foreground mask to compute the average HSV values
        average_hsv = cv2.mean(hsv_image, mask=foreground_mask)[:3]  # Only take H, S, V channels

        return average_hsv

    