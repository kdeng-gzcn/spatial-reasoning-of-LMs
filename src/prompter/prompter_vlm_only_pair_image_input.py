import numpy as np

from src.prompter.prompter_template import PromptTemplate
from src.utils import *
from src.vlm_only_prompts import (
    task_prompt_individual_vlm_zero_shot, 
    addtional_info_zero_shot, 
    VoT_zero_shot, 
    CoT_zero_shot,
    CoT_prompt,
    VoT_promopt,
)

class TaskPrompterVLMOnly(PromptTemplate):
    def __init__(self, **kwargs):
        super().__init__() # seed
        self.split = kwargs.get("split")
        self.is_shuffle = kwargs.get("is_shuffle")
        self.prompt_type = kwargs.get("prompt_type")

        self.short_dict = {
            0: "unable to judge",
            1: "leftward",
            2: "rightward",
            3: "no movement",
        }

        self.detailed_dict = {
            "leftward": "Leftward rotation – The camera rotated leftward horizontally.",
            "no movement": "No movement – The two images are completely identical with no even slight changes. This option should only be selected if there is absolute certainty that no movement at all as occurred.",
            "unable to judge": "Unable to judge – This option should only be selected in cases where the images are severely corrupted, fail to load, or lack sufficient visual information to determine movement.",
            "rightward": "Rightward rotation – The camera rotated rightward horizontally.",
        }

    def _shuffle_dict(self, dict: dict) -> dict:
        if self.is_shuffle:
            keys = list(dict.keys())
            np.random.shuffle(keys)
            new_dict = {i: dict[keys[i]] for i in range(len(keys))}
            return  new_dict
        else:
            return dict

    def __call__(self) -> str:
        option_map = self._shuffle_dict(self.short_dict) # short dict
        prompt = task_prompt_individual_vlm_zero_shot.format(
                opt1=self.detailed_dict[option_map[0]], 
                opt2=self.detailed_dict[option_map[1]],
                opt3=self.detailed_dict[option_map[2]],
                opt4=self.detailed_dict[option_map[3]],
            )
        
        if self.prompt_type == "zero-shot":
            prompt = prompt
        elif self.prompt_type == "add-info-zero-shot":
            prompt += addtional_info_zero_shot
        elif self.prompt_type == "VoT-zero-shot":
            prompt += addtional_info_zero_shot + VoT_zero_shot
        elif self.prompt_type == "CoT-zero-shot":
            prompt += addtional_info_zero_shot + CoT_zero_shot
        elif self.prompt_type == "CoT-prompt":
            prompt += addtional_info_zero_shot + CoT_prompt
        elif self.prompt_type == "VoT-prompt":
            prompt += addtional_info_zero_shot + VoT_zero_shot + VoT_promopt

        return prompt, option_map
    