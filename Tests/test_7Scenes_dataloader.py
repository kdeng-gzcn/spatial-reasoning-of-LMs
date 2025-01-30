import sys
sys.path.append("./")

from SpatialVLM.Dataset.SevenScenesDataset import SevenScenesImageDataset
from torch.utils.data import DataLoader

data_path = './data/Rebuild_7_Scenes_1200'  # set root data_path

dataset = SevenScenesImageDataset(data_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

count = 0

for batch in dataloader:

    if count >= 5:
        break

    source_image, target_image, metadata = batch
    print(metadata)
    print()

    count += 1
