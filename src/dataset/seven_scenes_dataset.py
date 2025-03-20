import os
import glob
import json

import torchvision.io as io
from torch.utils.data import Dataset

class SevenScenesImageDataset(Dataset):
    def __init__(self, data_root_dir=None, split=None):
        self.data_root_dir = data_root_dir
        self.subset = split

        self.data = []
        self._load_image_pairs(self.data_root_dir)

    def _load_image_pairs(self, data_dir=None):
        assert isinstance(data_dir, str), "Error in data_dir"
        if self.subset == "all":
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

        elif self.subset in ["tx", "ty", "tz", "theta", "phi", "psi"]:
            subset_path = os.path.join(self.data_root_dir, f"{self.subset}_significant")
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

                        self.data.append(pair_path)

        else:
            print(f"{self.subset} Not Recognized")

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
        metadata = None

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        item = {
            "source_image": source_image,
            "target_image": target_image,
            "metadata": metadata,
        }

        return item

if __name__ == "__main__":
    pass
        