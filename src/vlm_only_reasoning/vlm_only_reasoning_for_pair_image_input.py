import os
import json
import datetime
from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

from .vlm_only_reasoning_template import VLMOnlyReasoningTemplate

class VLMOnlyReasoning(VLMOnlyReasoningTemplate):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    def _make_results_dir(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.result_time_dir = os.path.join(self.result_dir, f"{current_time}")
        os.makedirs(self.result_time_dir, exist_ok=True)

        self.stat_dir = os.path.join(self.result_time_dir, "stat")
        os.makedirs(self.stat_dir, exist_ok=True)
        return self.result_time_dir
    
    def __call__(self):
        """run inference"""
        exp_config = {
            "data_root_dir": self.data_dir,
            "split": self.split,
            "is_shuffle": self.is_shuffle,
            "prompt_type": self.prompt_type,
            "VLM": self.vlm_id,
            "result_dir": self.result_dir,
        }

        new_history_list = []
        new_final_result_list = []

        dataloader_tqdm = tqdm(self.dataloader, desc="Processing", total=len(self.dataloader) if hasattr(self.dataloader, '__len__') else None)

        for batch in dataloader_tqdm:
            for item in batch:
                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]
                images = (source_image, target_image)

                self.VLM._clear_history() # clear the history of VLM for each pair of images

                task_prompt, opt_map = self.task_prompter() # __call__
                new_history_list.append({
                    "scene": metadata["scene"],
                    "seq": metadata["seq"],
                    "pair": metadata["pair"],
                    "label_dof": metadata["significance"],
                    "label": metadata["significance_text"],
                    "label_val": metadata["significance_value"],
                    "speaker": "User",
                    "receiver": "VLM",
                    "content": task_prompt,
                })

                VLM_answers = self.VLM.pipeline(images, task_prompt)
                new_history_list.append({
                    "scene": metadata["scene"],
                    "seq": metadata["seq"],
                    "pair": metadata["pair"],
                    "label_dof": metadata["significance"],
                    "label": metadata["significance_text"],
                    "label_val": metadata["significance_value"],
                    "speaker": "VLM",
                    "receiver": "User",
                    "content": VLM_answers,
                })

                pred = self.ans_parser(VLM_answers, metadata, mapping=opt_map)
                new_final_result_list.append({
                    "scene": metadata["scene"],
                    "seq": metadata["seq"],
                    "pair": metadata["pair"],
                    "label_dof": metadata["significance"],
                    "label": metadata["significance_text"],
                    "label_val": metadata["significance_value"],
                    "ans": pred["pred option"],
                    "pred": pred["pred text"],
                    "reason": pred["reason"],
                })

        try:
            self.VLM.print_total_tokens_usage()
        except:
            pass

        result_root_dir = Path(self._make_results_dir())

        config_json_path = result_root_dir / "config.json"
        history_csv_path = result_root_dir / "history.csv"
        inference_csv_path = result_root_dir / "inference.csv"

        json.dump(exp_config, open(config_json_path, "w"), indent=4)
        pd.DataFrame(new_history_list).to_csv(history_csv_path, index=False)
        pd.DataFrame(new_final_result_list).to_csv(inference_csv_path, index=False)
