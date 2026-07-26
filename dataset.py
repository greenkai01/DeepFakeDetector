"""
============================================================
dataset.py

Custom Dataset for DeepFake Detection

Author : YeonU Seo
============================================================
"""

from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config

# ----------------------------------------------------------
# Label
# ----------------------------------------------------------

CLASS_TO_IDX = {
    "real": 0,
    "fake": 1
}

# ----------------------------------------------------------
# Transform
# ----------------------------------------------------------

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((config.IMAGE_SIZE,
                       config.IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((config.IMAGE_SIZE,
                       config.IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

class DeepFakeDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.images = []
        self.labels = []

        self._load_dataset()


    def _load_dataset(self):

        for class_name in CLASS_TO_IDX:

            folder = self.root_dir / class_name

            if not folder.exists():
                continue

            for image_path in folder.iterdir():

                if image_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                ]:
                    continue

                self.images.append(image_path)
                self.labels.append(
                    CLASS_TO_IDX[class_name]
                )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):

        image_path = self.images[idx]

        label = self.labels[idx]

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Broken image skipped: {image_path}")

            return self.__getitem__(
                (idx + 1) % len(self.images)
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [label],
            dtype=torch.float32
        )

        return image, label