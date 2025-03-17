import random
import numpy as np
import torch

from src.prompter.utils import load_prompter # eyeprompt in dataloader
from src.dataset.utils import load_dataset # dataset
from torch.utils.data import DataLoader # dataloader
from src.Metric.utils import load_metric
from src.models.utils import load_model

class VLMOnlyReasoningTemplate():
    def __init__(self, **kwargs):

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        # Gloabl Config: multi-dict
        self.VLM_id = kwargs.get("VLM_id", None)
        self.data_path = kwargs.get("datapath", None)
        self.subset = kwargs.get("subset", None)
        self.result_dir = kwargs.get("result dir", None)
        self.is_shuffle = kwargs.get("is_shuffle", True)
        self.prompt_type = kwargs.get("prompt_type", "zero-shot")

        # 1. data
        dataset = load_dataset("7 Scenes", data_root_dir=self.data_path, split=self.subset)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        # 2.VLM
        self.VLM = load_model(self.VLM_id)
        self.VLM._load_weight()

        # 5. prompter
        self.task_prompter = load_prompter("Task Description for Baseline", 
                                           is_shuffle=self.is_shuffle, prompt_type=self.prompt_type)

        # 5. metric
        self.metric = load_metric("Baseline Metric 0123")

    def __call__(self):

        raise NotImplementedError()
