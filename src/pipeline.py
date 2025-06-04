from typing import Tuple, List, Any
import torch
import jsonlines
from pathlib import Path
import re

import logging

from src.prompt_generator import PromptGenerator

class SpatialReasoningPipeline:
    def __init__(self, config: Any, prompt_generator: PromptGenerator):
        self.config = config # global config
        self.logger = logging.getLogger(__name__)
        self.prompt_generator = prompt_generator
       
    def run_vlm_only(self, images: Tuple[torch.Tensor, torch.Tensor], vlm: Any, **kwargs):
        prompt = None
        vlm.pipeline(images, prompt)

    ###---------------multi-agents----------------###
    def _chat_history_saver_for_multi_agents(self, result_dir: str, **kwargs) -> None:
        chat_history_path = Path(result_dir) / "chat_history.jsonl"
        prompt = kwargs.get('prompt', None)
        vlm_answer = kwargs.get('vlm_answer', None)
        llm_answer = kwargs.get('llm_answer', None)
        metadata = kwargs.get('metadata', None)

        if vlm_answer:
            row = {
                **metadata,
                "prompt": prompt,
                "raw_output": vlm_answer,
                "model": "vlm",
                "round": self.idx,
            }
            with jsonlines.open(str(chat_history_path), mode='a') as writer:
                writer.write(row)

        if llm_answer:
            row = {
                **metadata,
                "prompt": prompt,
                "raw_output": llm_answer,
                "model": "llm",
                "round": self.idx,
            }
            with jsonlines.open(str(chat_history_path), mode='a') as writer:
                writer.write(row)
    
    def _result_saver_for_multi_agents(self, result_dir: str, **kwargs) -> None:
        pred = kwargs.get('pred', None)
        metadata = kwargs.get('metadata', None)
        result_path = Path(result_dir) / "inference.jsonl"
        row = {
            **metadata,
            "rsn": pred.get("rsn", None),
            "pred": pred.get("ans_text", None),
            "label": metadata.get("tx_text", None), # TODO: maybe phi_text
            "is_correct": pred.get("ans_text", None) == metadata.get("tx_text", None),
            "is_parse": pred.get("is_parse"),
            "round": self.idx,
        }
        with jsonlines.open(str(result_path), mode='a') as writer:
            writer.write(row)

    def _answer_parser_for_multi_agents(self, reasoning_answer: str, **kwargs) -> List[str]:
        option_map = kwargs.get('option_map', None)

        ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|$)", reasoning_answer, re.DOTALL)
        ques = ques_match.group(1) if ques_match else None
        if ques is not None:
            return {
                "ques": ques,
            }

        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", reasoning_answer, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else None

        ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", reasoning_answer, re.IGNORECASE)
        ans = int(ans_match.group(1)) if ans_match else None
        if ans is None or ans not in list(option_map.keys()): # avoid NoneType error
            logging.warning("Answer Option Not Extracted")
            return {
                "rsn": rsn,
                "ans": None,
                "ans_text": "Not extracted",
                "is_parse": False,  # indicate that the answer is not parsed correctly
            }

        return {
            "rsn": rsn,
            "ans": ans,
            "ans_text": option_map[ans],
            "is_parse": True,  # indicate that the answer is parsed correctly
        }

    def run_multi_agents(self, images: Tuple[torch.Tensor, torch.Tensor], vlm: Any, llm: Any, **kwargs) -> None:
        metadata = kwargs.get('metadata')
        self.idx = 0

        ###--------------------pipe-start---------------------###
        max_num_rounds = self.config.get('max_num_rounds', 10) # TODO: change with correct syntax
        result_dir = self.config.get('result_dir')
        is_save_result = kwargs.get('is_save_result', True)

        prompt_image_understanding = self.prompt_generator.image_caption_prompt(**kwargs)
        vlm_answer = vlm.pipeline(images, prompt_image_understanding)   
        if is_save_result:
            self._chat_history_saver_for_multi_agents(result_dir, vlm_answer=vlm_answer, metadata=metadata, prompt=prompt_image_understanding)

        for self.idx in range(max_num_rounds):
            if self.idx:
                if "ques" not in pred:
                    self.idx -= 1
                    break

                prompt_image_understanding = pred["ques"]
                vlm_answer = vlm.pipeline(images, prompt_image_understanding) 

                if is_save_result:
                    self._chat_history_saver_for_multi_agents(result_dir, vlm_answer=vlm_answer, metadata=metadata, prompt=prompt_image_understanding)

            prompt_spatial_reasoning, option_map = self.prompt_generator.spatial_reasoning_prompt_ma(vlm_answer=vlm_answer)
            llm_answer = llm.pipeline(prompt_spatial_reasoning)
            if is_save_result:
                self._chat_history_saver_for_multi_agents(result_dir, llm_answer=llm_answer, metadata=metadata, prompt=prompt_spatial_reasoning)

            pred = self._answer_parser_for_multi_agents(llm_answer, option_map=option_map)

        ### save final choice
        if is_save_result:
            self._result_saver_for_multi_agents(result_dir, metadata=metadata, pred=pred)
