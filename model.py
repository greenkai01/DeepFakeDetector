"""
============================================================
model.py

DeepFake Detector

Author : YeonU Seo
============================================================
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

import config


class DeepFakeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------
        # Load Pretrained EfficientNet
        # ---------------------------------------------------

        if config.PRETRAINED:

            backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT
            )

        else:

            backbone = efficientnet_b0(
                weights=None
            )

        # ---------------------------------------------------
        # Feature Extractor
        # ---------------------------------------------------

        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        feature_dim = backbone.classifier[1].in_features

        # ---------------------------------------------------
        # Freeze Backbone
        # ---------------------------------------------------

        if config.FREEZE_BACKBONE:

            for param in self.features.parameters():

                param.requires_grad = False

        # ---------------------------------------------------
        # Custom Classifier
        # ---------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(config.DROPOUT),

            nn.Linear(feature_dim,512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512,128),

            nn.ReLU(inplace=True),

            nn.Linear(128,1)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x,1)

        x = self.classifier(x)

        return x


# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


def print_model_summary(model):

    print("="*60)

    print(model)

    print("="*60)

    print(f"Trainable Parameters : {count_parameters(model):,}")

    print("="*60)


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__ == "__main__":

    model = DeepFakeDetector().to(config.DEVICE)

    print_model_summary(model)

    x = torch.randn(

        2,

        3,

        config.IMAGE_SIZE,

        config.IMAGE_SIZE

    ).to(config.DEVICE)

    y = model(x)

    print("Input Shape :",x.shape)

    print("Output Shape:",y.shape)