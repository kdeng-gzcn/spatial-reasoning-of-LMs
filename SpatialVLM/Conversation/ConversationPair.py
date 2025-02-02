from .ConversationTemplate import ConversationTemplate

import os
import json
import time

from tqdm import tqdm

class Conversations_Pairwise_Image(ConversationTemplate):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
    
    def __call__(self, len_conversation=1, result_dir=None):

        # setting up important part
        dataloader = self.dataloader # dataloader
        VLM = self.VLM # eyes after loading
        LLM = self.LLM # brain after loading
        LLM_prompter = self.LLM_prompter # prompter to help brain
        Task_prompter = self.start_prompter

        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        else:
            result_dir = "./Result/Pair Conversation Experiment/"
            os.makedirs(result_dir, exist_ok=True)

        data_count = 0

        result_json = [
            {
                "mode": "pair",
                "subset": self.subset,
                "VLM": VLM.model_name,
                "LLM": LLM.model_name,
            },
        ]

        dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)

        # for batch in dataloader:
        for batch in dataloader_tqdm:

            if data_count >= 60: # for small scale test
                    break
                
            data_count += 1

            # take 1 sample from 1 batch
            for item in batch:

                source_image = item["source_image"]
                target_image = item["target_image"]
                metadata = item["metadata"]

                # NOW WE GET A PAIR OF IMAGES and corresponding info
                # 0. Alg hyper params
                LLM.clearhistory() # for differnt pair images, clear up history in LLM
                len_conversation = len_conversation # max length of conversation
                images = (source_image, target_image) # fit in data structure

                # 1. load start prompt for task description
                task_prompt = Task_prompter(mode="pair")
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
                for idx in range(len_conversation): # for loop for max length or stop condition

                    if idx:
                        # if idx > 0 and LLM is not satisfied, then extract new questions for source
                        LLM_Questions_for_Both = LLM_Answer_for_round_idx

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
                    LLM_Questions_for_Both = "(Note that the first image is source image and the second one is target image) " + LLM_Questions_for_Both
                    VLM_Answer_for_Both = VLM.pipeline(images, LLM_Questions_for_Both)
                    VLM_Answer_for_Both = LLM_prompter.answer_prompter_pair(VLM_Answer_for_Both)

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
                            "choice": None,
                            "if continue conversation": None,
                            "answer": None
                        },
                    )

                # 3. evaluation with Metric.py
        
            # end of 1 item
            result_json.append(conversation_info)

        json_path = os.path.join(result_dir, f"{int(time.time())}.json")

        with open(json_path, "w") as f:

            json.dump(result_json, f, indent=4)
