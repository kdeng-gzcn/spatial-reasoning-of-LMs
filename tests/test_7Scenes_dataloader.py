import sys
sys.path.append("./")

from src.dataset.SevenScenesDataset import SevenScenesImageDataset
from torch.utils.data import DataLoader

data_path = './data/Rebuild_7_Scenes_1200'  # set root data_path

dataset = SevenScenesImageDataset(data_path, subset="phi")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

count = 0

# for batch in dataloader:
for batch in dataloader:

    if count >= 3: # for small scale test
            break
        
    count += 1

    # take 1 sample from 1 batch (edit dataloader later)
    for item in batch:

        source_image = item["source_image"]
        target_image = item["target_image"]
        metadata = item["metadata"]

        print(source_image)
        print()
        print(target_image)
        print()
        print(metadata)
        print
