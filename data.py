import numpy as np
import pandas as pd

from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torchvision.models import ResNet18_Weights

IMAGE_PATH_COL = "full_image_path"

class GlyphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_columns: List[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = str(row[IMAGE_PATH_COL])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Extract label values and safely convert to float32 numpy array
        vals = row[self.label_columns].values
        vals = pd.to_numeric(vals, errors="coerce") # type: ignore
        vals = np.nan_to_num(vals, nan=0.0).astype("float32")
        labels = torch.from_numpy(vals)   # shape: [num_labels]

        return image, labels


def load_dataframe_with_labels(csv_path: str):
    """
    Load the CSV and return:
        - df: cleaned DataFrame
        - label_columns: list of label column names
    """
    df = pd.read_csv(csv_path)

    # Columns that are NOT labels
    non_label_cols = [
        IMAGE_PATH_COL,
        "glyph_cote",
        "elements_original",
        "codex",
        "glyph_image",
        "Unnamed: 0",   # typical auto index column, if present
    ]
    non_label_cols = [c for c in non_label_cols if c in df.columns]

    label_columns = [c for c in df.columns if c not in non_label_cols]

    # Force labels to numeric float32
    df[label_columns] = (
        df[label_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
    )

    return df, label_columns


def build_default_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # ±15° rotation
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ], p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=None
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomGrayscale(p=0.5),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
