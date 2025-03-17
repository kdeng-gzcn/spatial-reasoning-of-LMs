from typing import Tuple

import numpy as np

from src.prompter.prompter_template import PromptTemplate # no use
from src.multi_agents_prompts import (
    task_prompt_zero_shot,
    dataset_prior_zero_shot,
    spatial_understanding_question_prompt_zero_shot,
    spatial_reasoning_prompt_zero_shot,
)

class TaskPrompterPairImageInput(PromptTemplate):
    def __init__(self, **kwargs):
        super().__init__() # no use
        self.prompt_type = kwargs.get("prompt_type")

    def __call__(self) -> str:
        prompt = task_prompt_zero_shot
        if self.prompt_type == "zero-shot":
            prompt = prompt
        elif self.prompt_type == "add-info-zero-shot":
            prompt += dataset_prior_zero_shot
        return prompt


class LLMQuestionToVLMPairImageInput(PromptTemplate):
    def __init__(self, **kwargs):
        super().__init__()
        self.prompt_type = kwargs.get("prompt_type")

    def __call__(self, llm_questions: str) -> str:
        prompt = spatial_understanding_question_prompt_zero_shot.format(llm_questions=llm_questions)
        if self.prompt_type == "zero-shot":
            prompt = prompt
        elif self.prompt_type == "add-info-zero-shot":
            prompt += dataset_prior_zero_shot
        return prompt


class VLMAnswerToLLMPairImageInput(PromptTemplate):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_shuffle = kwargs.get("is_shuffle")
        self.prompt_type = kwargs.get("prompt_type")

        seed = 42
        np.random.seed(seed)

        self.short_dict = {
            0: "ask more questions",
            1: "leftward",
            2: "rightward",
            3: "no movement",
        }

        self.detailed_dict = {
            "leftward": "Leftward rotation – The camera rotated leftward horizontally.",
            "no movement": "No movement – The two images are completely identical with no noticeable changes. This option should only be selected if there is absolute certainty that no movement at all has occurred.",
            "ask more questions": "Ask more questions – You are not confident to make a decision for now, and you want to ask VLM for more information.",
            "rightward": "Rightward rotation – The camera rotated rightward horizontally.",
        }

    def _shuffle_dict(self, dict: dict) -> dict:
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
    
    def __call__(self, vlm_answers: str) -> Tuple[str, dict]:
        option_map = self._shuffle_dict(self.short_dict) # short dict
        prompt =spatial_reasoning_prompt_zero_shot.format(
                vlm_answers=vlm_answers,
                opt1=self.detailed_dict[option_map[0]], 
                opt2=self.detailed_dict[option_map[1]],
                opt3=self.detailed_dict[option_map[2]],
                opt4=self.detailed_dict[option_map[3]],
            )
        
        if self.prompt_type == "zero-shot":
            prompt = prompt
        elif self.prompt_type == "add-info-zero-shot":
            prompt = prompt
        return prompt, option_map
