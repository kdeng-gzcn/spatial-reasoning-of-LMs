from src.dataset import (
    SevenScenesImageDataset,
)

def load_dataset(dataset_name: str, data_root_dir: str, split: str):
    dataset_mapping = {
        "7 Scenes": SevenScenesImageDataset,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["all", "tx", "ty", "tz", "theta", "phi", "psi"]:
        raise NotImplementedError(f"Subset {split} not supported.")

    return dataset_mapping[dataset_name](data_root_dir=data_root_dir, split=split)
