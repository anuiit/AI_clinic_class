# data.py
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
    """
    Simple multi-label dataset for codex glyphs.
    - df: DataFrame with:
        - IMAGE_PATH_COL column
        - label_columns: multi-hot 0/1 labels (one column per element).
    """

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


def build_default_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5),  # smaller & sane
            transforms.RandomGrayscale(p=0.5),      # keeps 3 channels
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
