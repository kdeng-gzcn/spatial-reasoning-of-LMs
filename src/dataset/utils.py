from src.dataset import (
    SevenScenesImageDataset,
    SevenScenesViewShiftDataset,
    ScanNetCameraMotionDataset,
    ScanNetViewShiftDataset,
    ScanNetppCameraMotionDataset,
)

def load_dataset(dataset_name: str, data_root_dir: str, **kwargs) -> object:
    dataset_mapping = {
        "single-dof-camera-motion-7-scenes": SevenScenesImageDataset,
        "obj-centered-view-shift-7-scenes": SevenScenesViewShiftDataset,

        "single-dof-camera-motion-scannet": ScanNetCameraMotionDataset,
        "obj-centered-view-shift-scannet": ScanNetViewShiftDataset,
        
        "single-dof-camera-motion-scannetpp": ScanNetppCameraMotionDataset,

        "obj-centered-view-shift-demo": ScanNetViewShiftDataset,
        "single-dof-camera-motion-demo": SevenScenesImageDataset,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")
    
    if kwargs.keys() in ["split"]:
        if kwargs.split not in ["all", "tx", "ty", "tz", "theta", "phi", "psi"]:
            raise NotImplementedError(f"Subset {kwargs.split} not supported.")

    return dataset_mapping[dataset_name](data_root_dir=data_root_dir, **kwargs)
