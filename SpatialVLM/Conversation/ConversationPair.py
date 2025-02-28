import os
import json
import time

import pandas as pd
from tqdm import tqdm

from .ConversationTemplate import ConversationTemplate

class Conversations_Pairwise_Image(ConversationTemplate):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
    
    def __call__(self):

        history_dict = [
            {
                "mode": "pair",
                "subset": self.subset,
                "VLM": self.VLM.model_name,
                "LLM": self.LLM.model_name,
            },
        ]

        dataloader_tqdm = tqdm(
            self.dataloader, 
            desc="Processing", 
            total=len(self.dataloader) if hasattr(self.dataloader, '__len__') else None
        )

        count = 0

        # for batch in dataloader:
        for batch in dataloader_tqdm:

            # if count >= 5:

            #     break

            # count += 1

            # take 1 sample from 1 batch
            for item in batch:

                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]

                conversation_info = [
                    {
                        "level": "Conversation Metadata",
                        "scene": metadata["scene"],
                        "seq": metadata["seq"],
                        "pair": metadata["pair"],
                        "significant dof": metadata["significance"],
                        "label": metadata["significance_text"],
                        "significant value": metadata["significance_value"],
                    },
                ]

                # NOW WE GET A PAIR OF IMAGES and corresponding info
                # 0. Alg hyper params
                self.LLM.clearhistory() # for differnt pair images, clear up history in LLM
                # len_conversation = len_conversation
                images = (source_image, target_image) # fit in data structure

                # 1. load start prompt for task description
                task_prompt = self.task_prompter()

                conversation_info.append(
                    {
                        "level": "round",
                        "round_num": 1,
                        "speaker": "User",
                        "listener": "LLM",
                        "text": task_prompt,
                    },
                )

                # 1 get the questions from LLM after it understand our task
                LLM_Questions_for_Both = self.LLM.pipeline(task_prompt)

                conversation_info.append(
                    {
                        "level": "round",
                        "round_num": 1,
                        "speaker": "LLM",
                        "listener": "VLM",
                        "text": LLM_Questions_for_Both
                    },
                )

                # 2. start conversation
                for idx in range(self.len_conv): # for loop for max length or stop condition

                    if idx:

                        if self.metric.option_map[self.metric.ans] != "ask more questions":

                            break

                        LLM_Questions_for_Both = self.metric.rsn + self.metric.ques

                        conversation_info.append(
                            {
                                "level": "round",
                                "round_num": idx + 1,
                                "speaker": "LLM",
                                "listener": "VLM",
                                "text": LLM_Questions_for_Both
                            },
                        )

                    # a. get answers from VLM given questions for source from LLM
                    LLM_Questions_for_Both = self.LLM2VLM_prompter(LLM_Questions=LLM_Questions_for_Both)

                    VLM_Answer_for_Both = self.VLM.pipeline(images, LLM_Questions_for_Both)
                    VLM_Answer_for_Both = self.VLM2LLM_prompter(VLM_Answers=VLM_Answer_for_Both)

                    conversation_info.append(
                        {
                            "level": "round",
                            "round_num": idx + 1,
                            "speaker": "VLM",
                            "listener": "LLM",
                            "text": VLM_Answer_for_Both
                        },
                    )

                    # b. now we want to get answers from LLM to see if he understand the images really
                    LLM_Answer_for_round_idx = self.LLM.pipeline(VLM_Answer_for_Both)

                    conversation_info.append(
                        {
                            "level": "end of round",
                            "round_num": idx + 1,
                            "speaker": "LLM",
                            "listener": "User or VLM",
                            "text": LLM_Answer_for_round_idx,
                        },
                    )

                    # 3. record the answer for this conversation
                    self.metric(
                        idx + 1, 
                        LLM_Answer_for_round_idx, 
                        metadata, 
                        mapping=self.VLM2LLM_prompter.option_map
                    )

            # end of 1 pair
            history_dict.append(conversation_info)

        """
        
        all data done, collect the stat
        
        """

        # end of entire alg for all dagtaset
        current_time = int(time.time())
        self.result_time_dir = os.path.join(self.result_dir, f"{current_time}")
        os.makedirs(self.result_time_dir, exist_ok=True)

        # 1. overall full conversation
        json_path = os.path.join(self.result_time_dir, f"conversations.json")
        with open(json_path, "w") as f:
            json.dump(history_dict, f, indent=4)

        # 2. overall csv/json result
        json_path = os.path.join(self.result_time_dir, f"result.json")
        with open(json_path, "w") as f:
            json.dump(self.metric.result_dict, f, indent=4)

        df = pd.DataFrame(self.metric.result_dict)
        csv_path = os.path.join(self.result_time_dir, f"result.csv")
        df.to_csv(csv_path, index=False)

        # 3. summary stat result
        stat_dicts = self.metric._evaluate()
        summary_stat =  stat_dicts["summary stat"]
        scalar_metrics_dict = summary_stat["scalar metrics"]
        cm_df = summary_stat["confusion matrix"]
        scalar_metrics_dict = {
            "mode": "pair",
            "subset": self.subset,
            "VLM": self.VLM.model_name,
            "LLM": self.LLM.model_name,
            **scalar_metrics_dict,
        }

        stat_dir = os.path.join(self.result_time_dir, "stat")
        os.makedirs(stat_dir, exist_ok=True)

        df = pd.DataFrame(scalar_metrics_dict)
        csv_path = os.path.join(stat_dir, f"summary_stat.csv")
        df.to_csv(csv_path, index=False)

        csv_path = os.path.join(stat_dir, f"summary_confusion_matrix.csv")
        cm_df.to_csv(csv_path)

        for round in range(1, len(stat_dicts)):
            
            single_round_stat_dict = stat_dicts[f"{round} round stat"]

            scalar_metrics_dict = single_round_stat_dict["scalar metrics"]
            cm_df = single_round_stat_dict["confusion matrix"]

            scalar_metrics_dict = {
                **scalar_metrics_dict,
            }

            stat_round_dir = os.path.join(stat_dir, f"round{round}")
            os.makedirs(stat_round_dir, exist_ok=True)

            df = pd.DataFrame(scalar_metrics_dict)
            csv_path = os.path.join(stat_round_dir, f"round{round}_stat.csv")
            df.to_csv(csv_path, index=False)

            csv_path = os.path.join(stat_round_dir, f"round{round}_confusion_matrix.csv")
            cm_df.to_csv(csv_path)
