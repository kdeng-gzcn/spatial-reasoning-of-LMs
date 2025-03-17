import os
import json
import datetime
from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

from .ConversationTemplate import MultiAgentsReasoningTemplate

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
            if count >= 20:
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

                full_history.append(
                    {
                        "scene": metadata["scene"],
                        "seq": metadata["seq"],
                        "pair": metadata["pair"],
                        "label_dof": metadata["significance"],
                        "label": metadata["significance_text"],
                        "label_val": metadata["significance_value"],
                        "idx": 1,
                        "speaker": "User",
                        "receiver": "LLM",
                        "content": task_prompt,
                    }
                )

                llm_questions_to_vlm = self.LLM.pipeline(task_prompt)

                for idx in range(self.max_len_of_conv): # for loop for max length or stop condition
                    if idx:
                        if pred["pred text"] != "ask more questions":
                            break

                        llm_questions_to_vlm = pred["question"]

                    llm_questions_to_vlm = self.spatial_question_prompter(llm_questions_to_vlm)

                    full_history.append(
                        {
                            "scene": metadata["scene"],
                            "seq": metadata["seq"],
                            "pair": metadata["pair"],
                            "label_dof": metadata["significance"],
                            "label": metadata["significance_text"],
                            "label_val": metadata["significance_value"],
                            "idx": idx+1,
                            "speaker": "LLM",
                            "receiver": "VLM",
                            "content": llm_questions_to_vlm,
                        }
                    )

                    vlm_answers_to_llm, opt_map = self.spatial_reasoning_prompter(
                        self.VLM.pipeline(images, llm_questions_to_vlm)
                    )

                    full_history.append(
                        {
                            "scene": metadata["scene"],
                            "seq": metadata["seq"],
                            "pair": metadata["pair"],
                            "label_dof": metadata["significance"],
                            "label": metadata["significance_text"],
                            "label_val": metadata["significance_value"],
                            "idx": idx+1,
                            "speaker": "VLM",
                            "receiver": "LLM",
                            "content": vlm_answers_to_llm,
                        }
                    )

                    llm_reasoning = self.LLM.pipeline(vlm_answers_to_llm)

                    full_history.append(
                        {
                            "scene": metadata["scene"],
                            "seq": metadata["seq"],
                            "pair": metadata["pair"],
                            "label_dof": metadata["significance"],
                            "label": metadata["significance_text"],
                            "label_val": metadata["significance_value"],
                            "idx": idx+1,
                            "speaker": "LLM",
                            "receiver": "User or VLM",
                            "content": llm_reasoning,
                        }
                    )

                    pred = self.parser(llm_reasoning, opt_map)

                    reasoning_result.append(
                        {
                            "scene": metadata["scene"],
                            "seq": metadata["seq"],
                            "pair": metadata["pair"],
                            "label_dof": metadata["significance"],
                            "label": metadata["significance_text"],
                            "label_val": metadata["significance_value"],
                            "idx": idx+1,
                            "pred_option": pred["pred option"],
                            "pred_text": pred["pred text"],
                            "reason": pred["reason"],
                            "question": pred["question"],
                        }
                    )

        result_root_dir = self._make_results_dir()
        config_json_path = result_root_dir / "config.json"
        history_csv_path = result_root_dir / "history.csv"
        inference_csv_path = result_root_dir / "inference.csv"

        json.dump(exp_config, open(config_json_path, "w"), indent=4)
        pd.DataFrame(full_history).to_csv(history_csv_path, index=False)
        pd.DataFrame(reasoning_result).to_csv(inference_csv_path, index=False)

        # """
        
        # all data done, collect the stat
        
        # """

        # # end of entire alg for all dagtaset
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # self.result_time_dir = os.path.join(self.result_dir, f"{current_time}")
        # os.makedirs(self.result_time_dir, exist_ok=True)

        # # 1. overall full conversation
        # json_path = os.path.join(self.result_time_dir, f"conversations.json")
        # with open(json_path, "w") as f:
        #     json.dump(history_dict, f, indent=4)

        # # 2. overall csv/json result
        # json_path = os.path.join(self.result_time_dir, f"result.json")
        # with open(json_path, "w") as f:
        #     json.dump(self.metric.result_dict, f, indent=4)

        # df = pd.DataFrame(self.metric.result_dict)
        # csv_path = os.path.join(self.result_time_dir, f"result.csv")
        # df.to_csv(csv_path, index=False)

        # # 3. summary stat result
        # stat_dicts = self.metric._evaluate()
        # summary_stat =  stat_dicts["summary stat"]
        # scalar_metrics_dict = summary_stat["scalar metrics"]
        # cm_df = summary_stat["confusion matrix"]
        # scalar_metrics_dict = {
        #     "mode": "pair",
        #     "subset": self.subset,
        #     "VLM": self.VLM.model_name,
        #     "LLM": self.LLM.model_name,
        #     **scalar_metrics_dict,
        # }

        # stat_dir = os.path.join(self.result_time_dir, "stat")
        # os.makedirs(stat_dir, exist_ok=True)

        # df = pd.DataFrame(scalar_metrics_dict)
        # csv_path = os.path.join(stat_dir, f"summary_stat.csv")
        # df.to_csv(csv_path, index=False)

        # csv_path = os.path.join(stat_dir, f"summary_confusion_matrix.csv")
        # cm_df.to_csv(csv_path)

        # for round in range(1, len(stat_dicts)):
            
        #     single_round_stat_dict = stat_dicts[f"{round} round stat"]

        #     scalar_metrics_dict = single_round_stat_dict["scalar metrics"]
        #     cm_df = single_round_stat_dict["confusion matrix"]

        #     scalar_metrics_dict = {
        #         **scalar_metrics_dict,
        #     }

        #     stat_round_dir = os.path.join(stat_dir, f"round{round}")
        #     os.makedirs(stat_round_dir, exist_ok=True)

        #     df = pd.DataFrame(scalar_metrics_dict)
        #     csv_path = os.path.join(stat_round_dir, f"round{round}_stat.csv")
        #     df.to_csv(csv_path, index=False)

        #     csv_path = os.path.join(stat_round_dir, f"round{round}_confusion_matrix.csv")
        #     cm_df.to_csv(csv_path)
