import numpy as np

from SpatialVLM.Prompter.PrompterTemplate import PromptTemplate
from SpatialVLM.prompts import task_prompt_individual_vlm_zero_shot, addtional_info_zero_shot, VoT_zero_shot, CoT_zero_shot

class TaskDesc_Prompter4Baseline(PromptTemplate):

    def __init__(self, **kwargs):
        self.split = None
        self.is_shuffle = kwargs.get("is_shuffle", False)
        self.prompt_type = kwargs.get("prompt_type", "zero-shot")
        
        seed = 42
        np.random.seed(seed)

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

        super().__init__()

    def _shuffle_dict(self, dict: dict):

        if self.is_shuffle:
            keys = list(dict.keys())
            np.random.shuffle(keys)
            new_dict = {
                0: dict[keys[0]],
                1: dict[keys[1]],
                2: dict[keys[2]],
                3: dict[keys[3]],
            }

            return  new_dict
        else:

            return dict

    def __call__(self):

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

        return prompt, option_map