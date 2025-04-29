import random
import numpy as np
import torch
import logging
from pathlib import Path

from src.prompter.utils import load_prompter # eyeprompt in dataloader
from src.dataset.utils import load_dataset # dataset
from torch.utils.data import DataLoader # dataloader
from src.ans_parser.utils import load_ans_parser
from src.models.utils import load_model

class VLMOnlyReasoningTemplate():
    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        self.vlm_id = kwargs.get("vlm_id")
        self.data_dir = Path(kwargs.get("data_dir"))
        self.split = kwargs.get("split")
        self.result_dir = kwargs.get("result_dir")
        self.is_shuffle = kwargs.get("is_shuffle")
        self.prompt_type = kwargs.get("prompt_type")

        dataset = load_dataset(self.data_dir.name, data_root_dir=self.data_dir, split=self.split)
        # check if the length is 0
        if dataset.__len__() == 0:
            raise ValueError(f"Dataset is empty. Please check the data directory: {self.data_dir}")
        else:
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        self.VLM = load_model(self.vlm_id)
        self.VLM._load_weight()

        self.task_prompter = load_prompter(
            "Task Prompt for VLM-Only Reasoning",
            is_shuffle=self.is_shuffle, 
            prompt_type=self.prompt_type,
            split=self.split,
        )

        self.ans_parser = load_ans_parser("Answer Parser for VLM-Only Reasoning")

    def __call__(self):
        raise NotImplementedError()
