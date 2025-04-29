import os
import glob
import json
import logging
from pathlib import Path

import torchvision.io as io
from torch.utils.data import Dataset

class SevenScenesImageDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = data_root_dir
        self.split = kwargs["split"]

        self.data = []
        self._load_image_pairs(self.data_root_dir)

    def _load_image_pairs(self, data_dir: str) -> None:
        if self.split == "all":
            for dof in os.listdir(self.data_root_dir):
                dof_path = os.path.join(self.data_root_dir, dof)
                if not os.path.isdir(dof_path):
                    continue 

                for scene in os.listdir(dof_path):
                    scene_path = os.path.join(dof_path, scene)
                    if not os.path.isdir(scene_path):
                        continue  

                    for seq in os.listdir(scene_path):
                        seq_path = os.path.join(scene_path, seq)
                        if not os.path.isdir(seq_path):
                            continue  

                        for pair in os.listdir(seq_path):
                            pair_path = os.path.join(seq_path, pair)
                            if not os.path.isdir(pair_path):
                                continue

                            self.data.append(pair_path)

        elif self.split in ["tx", "ty", "tz", "theta", "phi", "psi"]:
            subset_path = os.path.join(self.data_root_dir, f"{self.split}_significant")
            for scene in os.listdir(subset_path):
                scene_path = os.path.join(subset_path, scene)
                if not os.path.isdir(scene_path):
                    continue  

                for seq in os.listdir(scene_path):
                    seq_path = os.path.join(scene_path, seq)
                    if not os.path.isdir(seq_path):
                        continue  

                    for pair in os.listdir(seq_path):
                        pair_path = os.path.join(seq_path, pair)
                        if not os.path.isdir(pair_path):
                            continue

                        self.data.append(pair_path) # no need to count, max 90 pairs

        else:
            print(f"{self.split} Not Recognized")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair_path = self.data[idx]

        source_path = os.path.join(pair_path, "source")
        target_path = os.path.join(pair_path, "target")

        source_image_files = glob.glob(os.path.join(source_path, "*.color.png"))
        target_image_files = glob.glob(os.path.join(target_path, "*.color.png"))

        if not source_image_files or not target_image_files:
            raise FileNotFoundError(f"Missing images in {pair_path}")

        source_image_path = source_image_files[0]
        target_image_path = target_image_files[0]

        source_image = io.read_image(source_image_path)
        target_image = io.read_image(target_image_path)

        metadata_path = os.path.join(pair_path, "metadata.json")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        item = {
            "source_image": source_image,
            "target_image": target_image,
            "metadata": metadata,
        }

        return item
        

class SevenScenesViewShiftDataset(Dataset):
    def __init__(self, data_root_dir: str, **kwargs) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_root_dir = Path(data_root_dir)

        self.list_of_pair_path = []
        self.dataset_length_count = 0
        self._load_image_pairs()

    def _load_image_pairs(self) -> None:
        for scene_dir in self.data_root_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            for seq_dir in scene_dir.iterdir():
                if not seq_dir.is_dir():
                    continue

                for pair_dir in seq_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue

                    if self.dataset_length_count >= 250:
                        self.logger.warning(f"Dataset length count exceeded 250 for dir {self.data_root_dir}.")
                        return

                    self.list_of_pair_path.append(pair_dir)
                    self.dataset_length_count += 1

    def __len__(self):
        return len(self.list_of_pair_path)

    def __getitem__(self, idx):
        pair_path = self.list_of_pair_path[idx]

        source_image_path = pair_path / "source" / f"frame-{pair_path.name.split('-')[0]}.color.png"
        target_image_path = pair_path / "target" / f"frame-{pair_path.name.split('-')[1]}.color.png"

        source_image = io.read_image(source_image_path) # read image path as tensor
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