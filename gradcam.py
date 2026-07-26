"""
============================================================
gradcam.py

Grad-CAM Visualization

Author : YeonU Seo
============================================================
"""

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from model import DeepFakeDetector

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([

    transforms.Resize(

        (config.IMAGE_SIZE,

         config.IMAGE_SIZE)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

# ==========================================================
# Load Model
# ==========================================================

def load_model():

    model = DeepFakeDetector()

    model.load_state_dict(

        torch.load(

            config.BEST_MODEL,

            map_location=config.DEVICE

        )

    )

    model.to(config.DEVICE)

    model.eval()

    return model

# ==========================================================
# Create CAM
# ==========================================================

def create_gradcam(model):

    target_layers = [

        model.features[-1]

    ]

    cam = GradCAM(

        model=model,

        target_layers=target_layers

    )

    return cam

# ==========================================================
# Explain Image
# ==========================================================

def explain(image_path):

    model = load_model()

    cam = create_gradcam(model)

    image = Image.open(image_path).convert("RGB")

    image = image.resize(

        (config.IMAGE_SIZE,

         config.IMAGE_SIZE)

    )

    rgb = np.array(image)

    rgb_float = rgb.astype(np.float32) / 255

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(config.DEVICE)

    output = model(tensor)

    probability = torch.sigmoid(

        output

    ).item()

    target = [

        ClassifierOutputTarget(0)

    ]

    grayscale_cam = cam(

        input_tensor=tensor,

        targets=target

    )[0]

    visualization = show_cam_on_image(

        rgb_float,

        grayscale_cam,

        use_rgb=True

    )

    save_path = (

        config.FIGURE_DIR /

        "gradcam_result.png"

    )

    cv2.imwrite(

        str(save_path),

        cv2.cvtColor(

            visualization,

            cv2.COLOR_RGB2BGR

        )

    )

    print()

    print("="*60)

    print(f"Fake Probability : {probability:.4f}")

    print(f"Saved : {save_path}")

    print("="*60)

if __name__ == "__main__":

    explain(

        "sample.jpg"

    )