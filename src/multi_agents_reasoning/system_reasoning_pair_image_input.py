import os
import json
import datetime
from pathlib import Path
import logging
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from .system_reasoning_template import MultiAgentsReasoningTemplate

class MultiAgentsPairImageInputReasoning(MultiAgentsReasoningTemplate):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    def _make_results_dir(self) -> str:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_time_dir = Path(self.result_dir) / f"{current_time}"
        os.makedirs(result_time_dir, exist_ok=True)

        stat_dir = Path(result_time_dir) / "stat"
        os.makedirs(stat_dir, exist_ok=True)
        return result_time_dir
    
    def _full_history_append(self, full_history: List, metadata: Dict, idx: int, 
                             speaker: str, receiver: str, content: str) -> None:
        full_history.append(
            {
                "scene": metadata["scene"],
                "seq": metadata["seq"],
                "pair": metadata["pair"],
                "label_dof": metadata["significance"],
                "label": metadata["significance_text"],
                "label_val": metadata["significance_value"],
                "idx": idx,
                "speaker": speaker,
                "receiver": receiver,
                "content": content,
            }
        )

    def _reasoning_result_append(self, reasoning_result: List, metadata: Dict, idx: int, pred: Dict) -> None:
        reasoning_result.append(
            {
                "scene": metadata["scene"],
                "seq": metadata["seq"],
                "pair": metadata["pair"],
                "label_dof": metadata["significance"],
                "label": metadata["significance_text"],
                "label_val": metadata["significance_value"],
                "idx": idx,
                "pred_option": pred["pred option"],
                "pred_text": pred["pred text"],
                "reason": pred["reason"],
                "question": pred["question"],
            }
        )
    
    def __call__(self):
        exp_config = {
            "data_root_dir": self.data_dir,
            "split": self.split,
            "VLM": self.vlm_id,
            "vlm_image_input_type": self.vlm_image_input_type,
            "LLM": self.llm_id,
            "max_len_of_conv": self.max_len_of_conv,
            "prompt_type": self.prompt_type,
            "is_shuffle": self.is_shuffle,
            "result_dir": self.result_dir,
        }
        full_history = []
        reasoning_result = []
        dataloader_tqdm = tqdm(
            self.dataloader, 
            desc="Processing", 
            total=len(self.dataloader) if hasattr(self.dataloader, '__len__') else None
        )

        count = 0
        for batch in dataloader_tqdm:
            if count >=10:
                break
            count += 1
            
            for item in batch:
                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]
                images = (source_image, target_image)

                self.LLM._clear_history() # for differnt pair images, clear up history in LLM
                self.VLM._clear_history() # for differnt pair images, clear up history in VLM

                task_prompt = self.task_prompter() # __call__

                # full_history.append(
                #     {
                #         "scene": metadata["scene"],
                #         "seq": metadata["seq"],
                #         "pair": metadata["pair"],
                #         "label_dof": metadata["significance"],
                #         "label": metadata["significance_text"],
                #         "label_val": metadata["significance_value"],
                #         "idx": 1,
                #         "speaker": "User",
                #         "receiver": "LLM",
                #         "content": task_prompt,
                #     }
                # )

                self._full_history_append(full_history, metadata, 1, "User", "LLM", task_prompt)

                llm_questions_to_vlm = self.LLM.pipeline(task_prompt)

                for idx in range(self.max_len_of_conv): # for loop for max length or stop condition
                    if idx:
                        if pred["pred text"] != "ask more questions":
                            break

                        llm_questions_to_vlm = pred["reason"] + pred["question"]

                    llm_questions_to_vlm = self.spatial_question_prompter(llm_questions_to_vlm)

                    # full_history.append(
                    #     {
                    #         "scene": metadata["scene"],
                    #         "seq": metadata["seq"],
                    #         "pair": metadata["pair"],
                    #         "label_dof": metadata["significance"],
                    #         "label": metadata["significance_text"],
                    #         "label_val": metadata["significance_value"],
                    #         "idx": idx+1,
                    #         "speaker": "LLM",
                    #         "receiver": "VLM",
                    #         "content": llm_questions_to_vlm,
                    #     }
                    # )

                    self._full_history_append(full_history, metadata, idx+1, "LLM", "VLM", llm_questions_to_vlm)

                    vlm_answers_to_llm, opt_map = self.spatial_reasoning_prompter(
                        self.VLM.pipeline(images, llm_questions_to_vlm)
                    )

                    # full_history.append(
                    #     {
                    #         "scene": metadata["scene"],
                    #         "seq": metadata["seq"],
                    #         "pair": metadata["pair"],
                    #         "label_dof": metadata["significance"],
                    #         "label": metadata["significance_text"],
                    #         "label_val": metadata["significance_value"],
                    #         "idx": idx+1,
                    #         "speaker": "VLM",
                    #         "receiver": "LLM",
                    #         "content": vlm_answers_to_llm,
                    #     }
                    # )

                    self._full_history_append(full_history, metadata, idx+1, "VLM", "LLM", vlm_answers_to_llm)

                    llm_reasoning = self.LLM.pipeline(vlm_answers_to_llm)

                    # full_history.append(
                    #     {
                    #         "scene": metadata["scene"],
                    #         "seq": metadata["seq"],
                    #         "pair": metadata["pair"],
                    #         "label_dof": metadata["significance"],
                    #         "label": metadata["significance_text"],
                    #         "label_val": metadata["significance_value"],
                    #         "idx": idx+1,
                    #         "speaker": "LLM",
                    #         "receiver": "User or VLM",
                    #         "content": llm_reasoning,
                    #     }
                    # )

                    self._full_history_append(full_history, metadata, idx+1, "LLM", "User or VLM", llm_reasoning)

                    pred = self.parser(llm_reasoning, opt_map)

                    # reasoning_result.append(
                    #     {
                    #         "scene": metadata["scene"],
                    #         "seq": metadata["seq"],
                    #         "pair": metadata["pair"],
                    #         "label_dof": metadata["significance"],
                    #         "label": metadata["significance_text"],
                    #         "label_val": metadata["significance_value"],
                    #         "idx": idx+1,
                    #         "pred_option": pred["pred option"],
                    #         "pred_text": pred["pred text"],
                    #         "reason": pred["reason"],
                    #         "question": pred["question"],
                    #     }
                    # )

                    self._reasoning_result_append(reasoning_result, metadata, idx+1, pred)

        result_root_dir = self._make_results_dir()
        config_json_path = result_root_dir / "config.json"
        history_csv_path = result_root_dir / "history.csv"
        inference_csv_path = result_root_dir / "inference.csv"

        json.dump(exp_config, open(config_json_path, "w"), indent=4)
        pd.DataFrame(full_history).to_csv(history_csv_path, index=False)
        pd.DataFrame(reasoning_result).to_csv(inference_csv_path, index=False)
