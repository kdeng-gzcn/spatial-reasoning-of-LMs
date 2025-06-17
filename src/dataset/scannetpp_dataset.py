import os
import glob
import json
import logging
from pathlib import Path

import torchvision.io as io
from torch.utils.data import Dataset

class ScanNetppCameraMotionDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.cfg = kwargs.get("cfg")
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = data_root_dir
        self.dataset_length_count = 0

        self._load_image_pairs()

    def _load_image_pairs(self, data_dir: str) -> None:
        for scene in os.listdir(self.data_root_dir):
            hash_dir = os.path.join(self.data_root_dir, scene)
            if not os.path.isdir(hash_dir):
                continue  

            for pair in os.listdir(hash_dir):
                pair_dir = os.path.join(hash_dir, pair)
                if not os.path.isdir(pair_dir):
                    continue

                if self.dataset_length_count >= self.cfg.DATASET.UTILS.MAX_LEN_DATASET:
                    self.logger.warning(f"Dataset length count exceeded 60 for dir {self.data_root_dir}.")
                    return
                
                self.data.append(pair_dir)
                self.dataset_length_count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair_path = self.data[idx]
        source_path = pair_path / "source"
        target_path = pair_path / "target"

        source_image_path = next(source_path.glob("*.jpg"))
        target_image_path = next(target_path.glob("*.jpg"))

        source_image = io.read_image(source_image_path)
        target_image = io.read_image(target_image_path)

        metadata_path = pair_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        item = {
            "source_image": source_image,
            "target_image": target_image,
            "metadata": metadata,
        }

        return item
        