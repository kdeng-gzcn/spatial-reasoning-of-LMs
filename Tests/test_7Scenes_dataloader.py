import sys
sys.path.append("./")

from SpatialVLM.utils.dataset.dataloader_7Scenes import SevenScenesImageDataset
from torch.utils.data import DataLoader

data_path = './data/RGBD_7_Scenes_50_phi_15'  # set root data_path

dataset = SevenScenesImageDataset(root_dir=data_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

count = 0

for batch in dataloader:

    scene, seq, source_image_name, target_image_name, label_num, label_text, Description = batch
    print(scene, seq, source_image_name, target_image_name, label_num, label_text, Description)
    print()

    count += 1
    if count >= 5:
        break