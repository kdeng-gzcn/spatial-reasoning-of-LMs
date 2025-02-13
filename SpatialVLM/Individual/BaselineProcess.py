from .BaselineTemplate import IndividualTemplate

import os
import json
import time
import pandas as pd

from tqdm import tqdm

class IndividualProcess(IndividualTemplate):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _make_results_dir(self):

        current_time = int(time.time())

        self.result_time_dir = os.path.join(self.result_dir, f"{current_time}")
        os.makedirs(self.result_time_dir, exist_ok=True)

        self.stat_dir = os.path.join(self.result_time_dir, "stat")
        os.makedirs(self.stat_dir, exist_ok=True)
    
    def __call__(self):

        dataloader = self.dataloader
        VLM = self.VLM
        Task_prompter = self.task_prompter

        history_json = [
            {
                "subset": self.subset,
                "VLM": VLM.model_name,
            },
        ]

        dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

        data_count = 0

        # for batch in dataloader:
        for batch in dataloader_tqdm:

            # if data_count >= 20: # for small scale test
            #         break
                
            # data_count += 1

            # take 1 sample from 1 batch
            for item in batch:

                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]

                # NOW WE GET A PAIR OF IMAGES and corresponding info
                # 0. Alg hyper params
                images = (source_image, target_image) # fit in data structure

                # 1. load start prompt for task description
                task_prompt = Task_prompter()
                # 1.1 get the questions from LLM after it understand our task
                VLM_Answers = VLM.pipeline(images, task_prompt)

                each_pair_images_info = [
                    {
                        "level": "Metadata",
                        "scene": metadata["scene"],
                        "seq": metadata["seq"],
                        "pair": metadata["pair"],
                        "significant dof": metadata["significance"],
                        "label": metadata["significance_text"],
                        "significant value": metadata["significance_value"],
                    },
                    {
                        "level": "round",
                        "round_num": 1,
                        "speaker": "User",
                        "listener": "LLM",
                        "text": task_prompt,
                    },
                    {
                        "level": "round",
                        "round_num": 1,
                        "speaker": "LLM",
                        "listener": "VLM",
                        "text": VLM_Answers
                    },
                ]

                self.metric(VLM_Answers, metadata)

            history_json.append(each_pair_images_info)

        """
        
        all data done, collect the stat
        
        """

        self._make_results_dir()

        # 1. conversation
        json_path = os.path.join(self.result_time_dir, f"conversations.json")

        with open(json_path, "w") as f:

            json.dump(history_json, f, indent=4)

        # 2. result
        result_dict = self.metric.result_dict

        json_path = os.path.join(self.result_time_dir, f"result.json")

        with open(json_path, "w") as f:

            json.dump(result_dict, f, indent=4)

        df = pd.DataFrame(result_dict)
        csv_path = os.path.join(self.result_time_dir, f"result.csv")
        df.to_csv(csv_path, index=False)

        # 3. summary result
        stat_dicts = self.metric._evaluate()

        summary_stat =  stat_dicts["summary stat"]

        scalar_metrics_dict = summary_stat["scalar metrics"]
        confusion_matrix = summary_stat["confusion matrix"]

        scalar_metrics_dict = {
            "subset": self.subset,
            "VLM": VLM.model_name,
            **scalar_metrics_dict,
        }

        df = pd.DataFrame(scalar_metrics_dict)
        csv_path = os.path.join(self.stat_dir, f"summary_stat.csv")
        df.to_csv(csv_path, index=False)

        cm_df = pd.DataFrame(confusion_matrix, columns=["leftward", "rightward"], index=["leftward", "rightward"])
        csv_path = os.path.join(self.stat_dir, f"summary_confusion_matrix.csv")
        cm_df.to_csv(csv_path)
