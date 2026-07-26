"""
============================================================
predict.py

Predict DeepFake Image

Author : YeonU Seo
============================================================
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
from model import DeepFakeDetector

# ==========================================================
# Transform
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
# Face Detection
# ==========================================================

def detect_face(image):

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    detector = cv2.CascadeClassifier(

        cv2.data.haarcascades +

        "haarcascade_frontalface_default.xml"

    )

    faces = detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5

    )

    if len(faces) == 0:

        return image

    x, y, w, h = faces[0]

    face = image[

        y:y+h,

        x:x+w

    ]

    return face

# ==========================================================
# Predict
# ==========================================================

def predict(image_path):

    model = load_model()

    image = cv2.imread(str(image_path))

    if image is None:

        raise FileNotFoundError(image_path)

    face = detect_face(image)

    face = cv2.cvtColor(

        face,

        cv2.COLOR_BGR2RGB

    )

    pil = Image.fromarray(face)

    tensor = transform(pil)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(config.DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(

            output

        ).item()

    prediction = (

        "Fake"

        if probability >= config.THRESHOLD

        else

        "Real"

    )

    confidence = (

        probability

        if prediction == "Fake"

        else

        1 - probability

    )

    print()

    print("="*60)

    print(f"Prediction : {prediction}")

    print(f"Confidence : {confidence*100:.2f}%")

    print("="*60)

    return prediction, confidence

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    image_path = "sample.jpg"

    predict(image_path)