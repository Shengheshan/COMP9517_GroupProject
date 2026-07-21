from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class INatDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.data_root / row["relative_path"]
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
def load_dataframe(csv_path):
    data = pd.read_csv(csv_path)

    required_columns = {
        "relative_path",
        "label",
        "category_id",
        "species_name",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    return data