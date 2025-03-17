import random

import numpy as np
import torch

from src.prompter.utils import load_prompter
from src.dataset.utils import load_dataset
from torch.utils.data import DataLoader
from src.ans_parser.utils import load_metric
from src.models.utils import load_model

class MultiAgentsReasoningTemplate():
    def __init__(self, **kwargs):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.vlm_id = kwargs.get("vlm_id")
        self.llm_id = kwargs.get("llm_id")
        self.data_dir = kwargs.get("data_dir")
        self.split = kwargs.get("split")
        self.result_dir = kwargs.get("result_dir")
        self.is_shuffle = kwargs.get("is_shuffle")
        self.prompt_type = kwargs.get("prompt_type")
        self.max_len_of_conv = kwargs.get("max_len_of_conv")
        self.vlm_image_input_type = kwargs.get("vlm_image_input_type")

        dataset = load_dataset("7 Scenes", data_root_dir=self.data_dir, split=self.split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        self.VLM = load_model(self.vlm_id)
        self.VLM._load_weight()

        self.LLM = load_model(self.llm_id)
        self.LLM._load_weight()

        prompter_config = {
            "prompt_type": self.prompt_type,
            "is_shuffle": self.is_shuffle,
        }

        self.task_prompter = load_prompter("Task Prompt for Pair Image Input", **prompter_config)
        self.spatial_question_prompter = load_prompter("Spatial Understanding Question Prompt for Pair Image Input", **prompter_config)
        self.spatial_reasoning_prompter = load_prompter("Spatial Reasoning Prompt for Pair Image Input", **prompter_config)

        self.parser = load_metric("Reasoning Parser for Pair Image Input")

    def __call__(self):
        raise NotImplementedError()
