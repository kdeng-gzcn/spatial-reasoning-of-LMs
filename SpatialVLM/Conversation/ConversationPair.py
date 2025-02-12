from .ConversationTemplate import ConversationTemplate

import os
import json
import time
import pandas as pd

from tqdm import tqdm

class Conversations_Pairwise_Image(ConversationTemplate):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
    
    def __call__(self):

        dataloader = self.dataloader
        VLM = self.VLM
        LLM = self.LLM
        Task_prompter = self.task_prompter
        LLM_prompter = self.VLM2LLM_prompter
        VLM_prompter = self.LLM2VLM_prompter
        
        result_dir = self.result_dir
        len_conv = self.len_conv

        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        else:
            result_dir = "./Result/Pair Conversation Experiment/"
            os.makedirs(result_dir, exist_ok=True)

        conversation_json = [
            {
                "mode": "pair",
                "subset": self.subset,
                "VLM": VLM.model_name,
                "LLM": LLM.model_name,
            },
        ]

        dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

        data_count = 0

        # for batch in dataloader:
        for batch in dataloader_tqdm:

            # if data_count >= 60: # for small scale test
            #         break
                
            # data_count += 1

            # take 1 sample from 1 batch
            for item in batch:

                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]

                # NOW WE GET A PAIR OF IMAGES and corresponding info
                # 0. Alg hyper params
                LLM.clearhistory() # for differnt pair images, clear up history in LLM
                # len_conversation = len_conversation
                images = (source_image, target_image) # fit in data structure

                # 1. load start prompt for task description
                task_prompt = Task_prompter()
                # 1.1 get the questions from LLM after it understand our task
                LLM_Questions_for_Both = LLM.pipeline(task_prompt)

                conversation_info = [
                    {
                        "level": "Conversation Set Up",
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
                        "text": LLM_Questions_for_Both
                    },
                ]

                # 2. start conversation
                for idx in range(len_conv): # for loop for max length or stop condition

                    if idx:

                        if self.metric.ans != 0:  # Check if ans is not 0, then break the loop

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
                    LLM_Questions_for_Both = VLM_prompter(LLM_Questions=LLM_Questions_for_Both)

                    VLM_Answer_for_Both = VLM.pipeline(images, LLM_Questions_for_Both)
                    VLM_Answer_for_Both = LLM_prompter(VLM_Answers=VLM_Answer_for_Both)

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
                    LLM_Answer_for_round_idx = LLM.pipeline(VLM_Answer_for_Both)

                    conversation_info.append(
                        {
                            "level": "end of round",
                            "round_num": idx + 1,
                            "speaker": "LLM",
                            "listener": "User or VLM",
                            "text": LLM_Answer_for_round_idx,
                        },
                    )

                    # 3. evaluation end of 1 round
                    self.metric(idx + 1, LLM_Answer_for_round_idx, metadata)

            # end of 1 pair
            conversation_json.append(conversation_info)

        # end of entire alg for all dagtaset
        current_time = int(time.time())

        result_dir = os.path.join(result_dir, f"{current_time}")
        os.makedirs(result_dir, exist_ok=True)

        # 1. conversation
        json_path = os.path.join(result_dir, f"conversations.json")

        with open(json_path, "w") as f:

            json.dump(conversation_json, f, indent=4)

        # 2. result
        result_dict = self.metric.result_dict

        json_path = os.path.join(result_dir, f"result.json")

        with open(json_path, "w") as f:

            json.dump(result_dict, f, indent=4)

        df = pd.DataFrame(result_dict)
        csv_path = os.path.join(result_dir, f"result.csv")
        df.to_csv(csv_path, index=False)

        # 3. summary result
        stat_dicts = self.metric._evaluate()

        summary_stat =  stat_dicts["summary stat"]

        scalar_metrics_dict = summary_stat["scalar metrics"]
        confusion_matrix = summary_stat["confusion matrix"]

        scalar_metrics_dict = {
            "mode": "pair",
            "subset": self.subset,
            "VLM": VLM.model_name,
            "LLM": LLM.model_name,
            **scalar_metrics_dict,
        }

        stat_dir = os.path.join(result_dir, "stat")
        os.makedirs(stat_dir, exist_ok=True)

        df = pd.DataFrame(scalar_metrics_dict)
        csv_path = os.path.join(stat_dir, f"summary_stat.csv")
        df.to_csv(csv_path, index=False)

        cm_df = pd.DataFrame(confusion_matrix, columns=["leftward", "rightward"], index=["leftward", "rightward"])
        csv_path = os.path.join(stat_dir, f"summary_confusion_matrix.csv")
        cm_df.to_csv(csv_path)

        for round in range(1, len(stat_dicts)):
            
            stat_dict = stat_dicts[f"{round} round stat"]

            scalar_metrics_dict = stat_dict["scalar metrics"]
            confusion_matrix = stat_dict["confusion matrix"]

            scalar_metrics_dict = {
                **scalar_metrics_dict,
            }

            df = pd.DataFrame(scalar_metrics_dict)
            csv_path = os.path.join(stat_dir, f"round{round}_stat.csv")
            df.to_csv(csv_path, index=False)

            cm_df = pd.DataFrame(confusion_matrix, columns=["leftward", "rightward"], index=["leftward", "rightward"])
            csv_path = os.path.join(stat_dir, f"round{round}_confusion_matrix.csv")
            cm_df.to_csv(csv_path)
