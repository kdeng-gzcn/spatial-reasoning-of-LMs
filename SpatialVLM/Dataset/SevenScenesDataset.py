import os
from PIL import Image
import glob
import json

from torchvision import transforms
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader


# import sys
# sys.path.append("./")

class SevenScenesImageDataset(Dataset):

    def __init__(self, data_root_dir):
        """
        
        input:
            dataset path in string
        
        """

        self.data_root_dir = data_root_dir

        self.data = []
        self._load_image_pairs(self.data_root_dir)

    def _load_image_pairs(self, data_dir=None):

        assert isinstance(data_dir, str), "Error in data_dir"

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

                        # source_path = os.path.join(pair_path, "source")
                        # target_path = os.path.join(pair_path, "target")

                        # source_image_path = glob.glob(source_path, "*.png")
                        # source_image_path = source_image_path[0]
                        # target_image_path = glob.glob(target_path, "*.png")
                        # target_image_path = target_image_path[0]

                        # self.data.append((source_image_path, target_image_path))

                        self.data.append(pair_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pair_path = self.data[idx]

        source_path = os.path.join(pair_path, "source")
        target_path = os.path.join(pair_path, "target")

        source_image_files = glob.glob(os.path.join(source_path, "*.png"))
        target_image_files = glob.glob(os.path.join(target_path, "*.png"))

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

        return source_image, target_image, metadata

if __name__ == "__main__":

    pass
        