import yaml
import numpy as np

from src.prompter.prompter_template import PromptTemplate
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
        self.options_config = yaml.safe_load(open("src/vlm_only_options.yaml", "r"))
        self.split = kwargs.get("split")
        self.options_config = self.options_config.get("directions").get(self.split)
        self.is_shuffle = kwargs.get("is_shuffle")
        self.prompt_type = kwargs.get("prompt_type")

        self.short_dict = self.options_config.get("short_dict")
        self.detailed_dict = self.options_config.get("detailed_dict")

        self.addtional_info_zero_shot = addtional_info_zero_shot.format(additional_info=self.options_config.get("additional_info"))
        self.CoT_zero_shot = CoT_zero_shot.format(CoT_reasoning_skills=self.options_config.get("CoT"))
        self.VoT_zero_shot = VoT_zero_shot.format(VoT_reasoning_skills=self.options_config.get("VoT"))

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
            prompt += self.addtional_info_zero_shot
        elif self.prompt_type == "VoT-zero-shot":
            prompt += self.addtional_info_zero_shot + self.VoT_zero_shot
        elif self.prompt_type == "CoT-zero-shot":
            prompt += self.addtional_info_zero_shot + self.CoT_zero_shot
        # elif self.prompt_type == "CoT-prompt":
        #     prompt += self.addtional_info_zero_shot + CoT_prompt
        # elif self.prompt_type == "VoT-prompt":
        #     prompt += self.addtional_info_zero_shot + self.VoT_zero_shot + VoT_promopt

        return prompt, option_map
    