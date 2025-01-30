from SpatialVLM.Dataset import (
    SevenScenesImageDataset,
)

def load_dataset(dataset_name, data_root_dir, split=None):

    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        name (str, optional): The dataset configuration to load.
    """

    dataset_mapping = {
        "7 Scenes": SevenScenesImageDataset,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test", None]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](data_root_dir=data_root_dir)
