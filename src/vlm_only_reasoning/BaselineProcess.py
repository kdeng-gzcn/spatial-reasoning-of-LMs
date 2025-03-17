import os
import json
import datetime
from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

from .BaselineTemplate import VLMOnlyReasoningTemplate

class VLMOnlyReasoning(VLMOnlyReasoningTemplate):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    def _make_results_dir(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. result_dir with time
        self.result_time_dir = os.path.join(self.result_dir, f"{current_time}")
        os.makedirs(self.result_time_dir, exist_ok=True)

        # 2. result stat dir
        self.stat_dir = os.path.join(self.result_time_dir, "stat")
        os.makedirs(self.stat_dir, exist_ok=True)

        return self.result_time_dir
    
    def __call__(self):
        """run inference"""
        exp_config = {
            "data_root_dir": self.data_path,
            "split": self.subset,
            "is_shuffle": self.is_shuffle,
            "prompt_type": self.prompt_type,
            "VLM": self.VLM_id,
            "result_dir": self.result_dir,
        }
        
        history_json = []
        new_history_dict = {
            "scene": [],
            "seq": [],
            "pair": [],
            "label_dof": [],
            "label": [],
            "label_deg": [],
            "speaker": [],
            "receiver": [], 
            "content": [],
        }

        new_final_result_dict = {
            "scene": [],
            "seq": [],
            "pair": [],
            "label_dof": [],
            "label": [],
            "label_deg": [],
            "ans": [],
            "pred": [],
            "reason": [],
        }

        dataloader_tqdm = tqdm(self.dataloader, desc="Processing", total=len(self.dataloader) if hasattr(self.dataloader, '__len__') else None)

        for batch in dataloader_tqdm:
            for item in batch:
                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]
                images = (source_image, target_image)

                self.VLM._clear_history()
                task_prompt, opt_map = self.task_prompter() # __call__

                keys = [
                    "scene", "seq", "pair", 
                    "label_dof", "label", "label_deg", 
                    "speaker", "receiver", "content"
                ]

                values = [
                    metadata["scene"], 
                    metadata["seq"], metadata["pair"], 
                    metadata["significance"], 
                    metadata["significance_text"], 
                    metadata["significance_value"],
                    "User", "VLM", task_prompt
                ]

                for key, value in zip(keys, values):
                    new_history_dict[key].append(value)

                # 1.1 get the questions from LLM after it understand our task
                VLM_answers = self.VLM.pipeline(images, task_prompt)
                self.VLM._clear_history()

                keys = [
                    "scene", "seq", "pair", 
                    "label_dof", "label", "label_deg", 
                    "speaker", "receiver", "content"
                ]

                values = [
                    metadata["scene"], 
                    metadata["seq"], metadata["pair"], 
                    metadata["significance"], 
                    metadata["significance_text"], 
                    metadata["significance_value"],
                    "VLM", "User", VLM_answers
                ]

                for key, value in zip(keys, values):
                    new_history_dict[key].append(value)

                # each_pair_images_info = [
                #     {
                #         "level": "metadata",
                #         "scene": metadata["scene"],
                #         "seq": metadata["seq"],
                #         "pair": metadata["pair"],
                #         "significant dof": metadata["significance"],
                #         "label": metadata["significance_text"],
                #         "significant value": metadata["significance_value"],
                #     },
                #     {
                #         "level": "round",
                #         "round_num": 1,
                #         "speaker": "User",
                #         "listener": "LLM",
                #         "text": task_prompt,
                #     },
                #     {
                #         "level": "round",
                #         "round_num": 1,
                #         "speaker": "LLM",
                #         "listener": "VLM",
                #         "text": VLM_answers
                #     },
                # ]

                pred = self.metric(VLM_answers, metadata, mapping=opt_map)

                keys = list(new_final_result_dict.keys())

                values = [
                    metadata["scene"], 
                    metadata["seq"], metadata["pair"], 
                    metadata["significance"], 
                    metadata["significance_text"], 
                    metadata["significance_value"],
                    pred["pred option"], pred["pred text"], pred["reason"]
                ]

                for key, value in zip(keys, values):
                    new_final_result_dict[key].append(value)

            # history_json.append(each_pair_images_info)

        result_root_dir = Path(self._make_results_dir())

        config_json_path = result_root_dir / "config.json"
        history_csv_path = result_root_dir / "history.csv"
        inference_csv_path = result_root_dir / "inference.csv"

        json.dump(exp_config, open(config_json_path, "w"), indent=4)
        pd.DataFrame(new_history_dict).to_csv(history_csv_path, index=False)
        pd.DataFrame(new_final_result_dict).to_csv(inference_csv_path, index=False)

        # # 1. conversation
        # json_path = os.path.join(self.result_time_dir, f"conversations.json")
        # with open(json_path, "w") as f:
        #     json.dump(history_json, f, indent=4)

        # # 2. result dict to json/csv
        # result_dict = self.metric.result_dict

        # json_path = os.path.join(self.result_time_dir, f"result.json")
        # with open(json_path, "w") as f:
        #     json.dump(result_dict, f, indent=4)

        # df = pd.DataFrame(result_dict)
        # csv_path = os.path.join(self.result_time_dir, f"result.csv")
        # df.to_csv(csv_path, index=False)

        # # 3. summary result dict to csv
        # stat_dicts = self.metric._evaluate()

        # summary_stat =  stat_dicts["summary stat"]

        # scalar_metrics_dict = summary_stat["scalar metrics"]
        # cm_df = summary_stat["confusion matrix"]

        # scalar_metrics_dict = {
        #     "subset": self.subset,
        #     "VLM": self.VLM.model_name,
        #     **scalar_metrics_dict,
        # }

        # df = pd.DataFrame(scalar_metrics_dict)
        # csv_path = os.path.join(self.stat_dir, f"summary_stat.csv")
        # df.to_csv(csv_path, index=False)

        # csv_path = os.path.join(self.stat_dir, f"summary_confusion_matrix.csv")
        # cm_df.to_csv(csv_path)
