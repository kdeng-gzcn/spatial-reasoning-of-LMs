from typing import Tuple, List, Any
import torch
import jsonlines
from pathlib import Path
import re

import logging

class SpatialReasoningPipeline:
    def __init__(self, config: dict):
        self.config = config # global config
        self.logger = logging.getLogger(__name__)
       
    def run_vlm_only(self, images: Tuple[torch.Tensor, torch.Tensor], vlm: Any, **kwargs):
        prompt = None
        vlm.pipeline(images, prompt)

    def _chat_history_saver_for_multi_agents(self, result_dir: str, **kwargs) -> None:
        chat_history_path = Path(result_dir) / "chat_history.jsonl"
        vlm_answer = kwargs.get('vlm_answer', None)
        llm_answer = kwargs.get('llm_answer', None)
        metadata = kwargs.get('metadata', None)

        if vlm_answer:
            with jsonlines.open(str(chat_history_path), mode='a') as writer:
                writer.write(vlm_answer)

        if llm_answer:
            with jsonlines.open(str(chat_history_path), mode='a') as writer:
                writer.write(llm_answer)
    
    def _result_saver_for_multi_agents(self, result_dir: str, **kwargs) -> None:
        result_path = Path(result_dir) / "inference.jsonl"
        # TODO:

    def _answer_parser_for_multi_agents(self, reasoning_answer: str, **kwargs) -> List[str]:
        # TODO: make nicer function together with prompt candidate generator???
        ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|$)", text, re.DOTALL)
        ques = ques_match.group(1) if ques_match else None
        if ques is not None:
            return {
                "ques": ques,
            }

        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", text, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else "None"

        ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", text, re.IGNORECASE)
        ans = int(ans_match.group(1)) if ans_match else None
        if ans is None or ans not in [0, 1, 2, 3]: # avoid NoneType error
            logging.warning("Answer Option Not Extracted")
            ans = next(key for key, value in option_map.items() if value == "unable to judge")

        return {
            "rsn": rsn,
            "ans": ans,
            "ans_text": option_map[ans],
        }

    def run_multi_agents(self, images: Tuple[torch.Tensor, torch.Tensor], vlm: Any, llm: Any, **kwargs) -> None:
        metadata = kwargs.get('metadata')

        prompt_image_understanding = None # for inital caption, also for later consersational spatial question
        prompt_spatial_reasoning = None

        max_num_rounds = self.config.get('max_num_rounds') # TODO: change with correct syntax
        result_dir = self.config.get('result_dir') # TODO: change with correct syntax
        is_save_result = kwargs.get('is_save_result', False) # default to False

        vlm_answer = vlm.pipeline(images, prompt_image_understanding)
        self._chat_history__saver_for_multi_agents(result_dir, vlm_answer=vlm_answer, metadata=metadata) if is_save_result else None

        for idx in range(max_num_rounds):
            if idx:
                # TODO: change with correct syntax

            llm_answer = llm.pipeline(vlm_answer, prompt_spatial_reasoning)
            self._chat_history__saver_for_multi_agents(result_dir, llm_answer=llm_answer, metadata=metadata) if is_save_result else None

            pred = self._answer_parser_for_multi_agents(llm_answer, option_map=kwargs.get('option_map', {})) # TODO: change with correct syntax





