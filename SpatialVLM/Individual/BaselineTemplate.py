import random
import numpy as np
import torch

# load prompter
from SpatialVLM.Prompter.utils import load_prompter # eyeprompt in dataloader
# load dataset
from SpatialVLM.Dataset.utils import load_dataset # dataset
from torch.utils.data import DataLoader # dataloader
# load metrics
from SpatialVLM.Metric.utils import load_metric
# load model
from SpatialVLM.Model.utils import load_model

class IndividualTemplate():

    def __init__(self, **kwargs):

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        # Gloabl Config: multi-dict
        self.VLM_id = kwargs.get("VLM_id", None)
        data_path = kwargs.get("datapath", None)
        self.subset = kwargs.get("subset", None)
        self.result_dir = kwargs.get("result dir", None)

        # 1. data
        dataset = load_dataset("7 Scenes", data_root_dir=data_path, subset=self.subset)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        # 2.VLM
        self.VLM = load_model("Phi 3.5")
        self.VLM._load_weight()

        # 5. prompter
        self.task_prompter = load_prompter("Task Description for Baseline")

        # 5. metric
        self.metric = load_metric("Baseline Metric 0123")

    def __call__(self):

        raise NotImplementedError()
