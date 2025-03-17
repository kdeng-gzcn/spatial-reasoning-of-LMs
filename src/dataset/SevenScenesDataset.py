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

            subset_path = os.path.join(self.data_root_dir, f"{self.subset}_Significant")

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


# class SevenScenesImageDataset(Dataset):
#     """
#     Dataset for loading paired images and metadata from SevenScenes dataset.
#     """

#     def __init__(self, data_root_dir=None, subset=None):
#         assert isinstance(data_root_dir, str) and os.path.isdir(data_root_dir), "Error: data_root_dir must be a valid directory path."
#         self.data_root_dir = data_root_dir
#         self.subset = subset

#         # Store valid subsets in a set for faster lookup
#         self.valid_subsets = {"tx", "ty", "tz", "theta", "phi", "psi"}

#         # Initialize data list
#         self.data = []
#         self._load_image_pairs()

#     def _load_image_pairs(self):
#         """
#         Loads the paths to image pairs based on subset selection.
#         """
#         if self.subset is None:
#             for dof_entry in os.scandir(self.data_root_dir):  # Using os.scandir() for better performance
#                 if not dof_entry.is_dir():
#                     continue
#                 self._process_scene(dof_entry.path)

#         elif self.subset in self.valid_subsets:
#             subset_path = os.path.join(self.data_root_dir, f"{self.subset}_Significant")
#             if not os.path.isdir(subset_path):
#                 raise ValueError(f"Subset path does not exist: {subset_path}")

#             self._process_scene(subset_path)

#     def _process_scene(self, root_path):
#         """
#         Recursively processes all scenes, sequences, and image pairs.
#         """
#         for scene_entry in os.scandir(root_path):
#             if not scene_entry.is_dir():
#                 continue

#             for seq_entry in os.scandir(scene_entry.path):
#                 if not seq_entry.is_dir():
#                     continue

#                 for pair_entry in os.scandir(seq_entry.path):
#                     if pair_entry.is_dir():
#                         self.data.append(pair_entry.path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         """
#         Loads source and target images along with their metadata.
#         """
#         pair_path = self.data[idx]

#         # Load images
#         source_image = self._load_image(os.path.join(pair_path, "source"))
#         target_image = self._load_image(os.path.join(pair_path, "target"))

#         # Load metadata
#         metadata = self._load_metadata(os.path.join(pair_path, "metadata.json"))

#         return {"source_image": source_image, "target_image": target_image, "metadata": metadata}

#     def _load_image(self, image_dir):
#         """
#         Loads an image from the given directory.
#         """
#         image_files = glob.glob(os.path.join(image_dir, "*.png"))
#         if not image_files:
#             raise FileNotFoundError(f"No images found in {image_dir}")
#         if len(image_files) > 1:
#             print(f"Warning: Multiple images found in {image_dir}. Using the first image.")

#         try:
#             return io.read_image(image_files[0])
#         except Exception as e:
#             raise IOError(f"Failed to read image {image_files[0]}: {e}")

#     def _load_metadata(self, metadata_path):
#         """
#         Loads metadata from a JSON file, handling errors.
#         """
#         if not os.path.exists(metadata_path):
#             return None

#         try:
#             with open(metadata_path, "r") as f:
#                 return json.load(f)
#         except (json.JSONDecodeError, IOError) as e:
#             print(f"Warning: Failed to read metadata {metadata_path}: {e}")
#             return None


if __name__ == "__main__":

    pass
        