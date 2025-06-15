import json
import logging
from pathlib import Path

import torchvision.io as io
from torch.utils.data import Dataset

class ScanNetCameraMotionDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = Path(data_root_dir)
        if kwargs["split"]:
            self.split = kwargs["split"]
        else:
            self.logger.warning("No split provided, defaulting to 'all'")
            self.split = "all"

        self.data = []
        self.dataset_length_count = 0
        self._load_image_pairs(self.data_root_dir)

    def _load_image_pairs(self, data_dir: Path) -> None:
        if self.split == "all":
            for dof in data_dir.iterdir():
                if not dof.is_dir():
                    continue

                for scene in dof.iterdir():
                    if not scene.is_dir():
                        continue

                    for pair in scene.iterdir():
                        if not pair.is_dir():
                            continue

                        self.data.append(pair)

        elif self.split in ["tx", "ty", "tz", "theta", "phi", "psi"]:
            split_path = data_dir / f"{self.split}_significant"
            if not split_path.exists():
                self.logger.error(f"Split path {split_path} does not exist.")
                return
            
            for scene in split_path.iterdir():
                if not scene.is_dir():
                    continue

                for pair in scene.iterdir():
                    if not pair.is_dir():
                        continue

                    if self.dataset_length_count >= 150:
                        self.logger.warning(f"Dataset length count exceeded 150 for {self.split} split.")
                        return

                    self.data.append(pair)
                    self.dataset_length_count += 1

        else:
            self.logger.error(f"Invalid split: {self.split}. Must be 'all' or one of the DOFs.")

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

                if self.dataset_length_count >= self.cfg.DATASET.UTILS.MAX_LEN_DATASET: # TODO: set in config
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