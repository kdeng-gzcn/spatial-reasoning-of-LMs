import json
import logging
from pathlib import Path

import torchvision.io as io
from torch.utils.data import Dataset

class ScanNetCameraMotionDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.cfg = kwargs.get("cfg")
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = Path(data_root_dir)

        self.list_of_pair_path = []
        self.dataset_length_count = 0
        self._load_image_pairs()

    def _load_image_pairs(self) -> None:
        for scene_dir in self.data_root_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            for pair_dir in scene_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                
                if self.cfg: 
                    if self.dataset_length_count >= self.cfg.DATASET.UTILS.MAX_LEN_DATASET:
                        self.logger.warning(f"Dataset length count exceeded {self.cfg.DATASET.UTILS.MAX_LEN_DATASET} for dir {self.data_root_dir}.")
                        return

                self.list_of_pair_path.append(pair_dir)
                self.dataset_length_count += 1

    def __len__(self):
        return len(self.list_of_pair_path)

    def __getitem__(self, idx):
        pair_path = self.list_of_pair_path[idx]
        source_path = pair_path / "source"
        target_path = pair_path / "target"

        try:
            source_image_path = next(source_path.glob("*.jpg"))
            target_image_path = next(target_path.glob("*.jpg"))
        except Exception as e:
            self.logger.error(f"Error finding image files in {pair_path}: {e}")

        source_image = io.read_image(source_image_path)
        target_image = io.read_image(target_image_path)

        metadata_path = pair_path / "metadata.json"
        try: 
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading metadata file {metadata_path}: {e}")

        item = {
            "source_image": source_image,
            "target_image": target_image,
            "metadata": metadata,
        }

        return item
        

class ScanNetViewShiftDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.cfg = kwargs.get("cfg")
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = Path(data_root_dir)

        self.list_of_pair_path = []
        self.dataset_length_count = 0
        self._load_image_pairs()

    def _load_image_pairs(self) -> None:
        for scene_dir in self.data_root_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            for pair_dir in scene_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                
                if self.cfg:
                    if self.dataset_length_count >= self.cfg.DATASET.UTILS.MAX_LEN_DATASET:
                        self.logger.warning(f"Dataset length count exceeded 60 for dir {self.data_root_dir}.")
                        return

                self.list_of_pair_path.append(pair_dir)
                self.dataset_length_count += 1

    def __len__(self):
        return len(self.list_of_pair_path)

    def __getitem__(self, idx):
        pair_path = self.list_of_pair_path[idx]
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