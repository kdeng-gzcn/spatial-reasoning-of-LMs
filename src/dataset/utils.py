from src.dataset import (
    SevenScenesImageDataset,
    SevenScenesRelativePoseDataset,
    ScanNetSpatialReasoningDataset,
    ScanNetYawDataset,
    ScanNetppSpatialReasoningDataset,
)

def load_dataset(dataset_name: str, data_root_dir: str, **kwargs) -> object:
    dataset_mapping = {
        "7 Scenes": SevenScenesImageDataset,
        "relative-pose-7-scenes": SevenScenesRelativePoseDataset,
        "spatial-reasoning-scannet": ScanNetSpatialReasoningDataset,
        "relative-pose-scannet": ScanNetYawDataset,
        "spatial-reasoning-scannetpp": ScanNetppSpatialReasoningDataset,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")
    
    if kwargs.keys() in ["split"]:
        if kwargs.split not in ["all", "tx", "ty", "tz", "theta", "phi", "psi"]:
            raise NotImplementedError(f"Subset {kwargs.split} not supported.")

    return dataset_mapping[dataset_name](data_root_dir=data_root_dir, **kwargs)
