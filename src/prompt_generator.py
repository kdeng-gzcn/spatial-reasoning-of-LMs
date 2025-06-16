import random

from typing import Any

camera_pose_list = [
    "left", "right", "leftward", "rightward",
    "up", "down", "upward", "downward",
    "forward", "backward", "clockwise", "counterclockwise",
]

camera_pose_explanation = {
    "left": "Leftward translation – The camera moved leftward horizontally.",
    "right": "Rightward translation – The camera moved rightward horizontally.",
    "leftward": "Leftward rotation – The camera rotated leftward horizontally.",
    "rightward": "Rightward rotation – The camera rotated rightward horizontally.",
    "up": "Upward translation – The camera moved upward vertically.",
    "down": "Downward translation – The camera moved downward vertically.",
    "upward": "Upward rotation – The camera rotated upward vertically.",
    "downward": "Downward rotation – The camera rotated downward vertically.",
    "forward": "Forward translation – The camera moved forward along the viewing direction.",
    "backward": "Backward translation – The camera moved backward along the viewing direction.",
    "clockwise": "Clockwise rotation – The camera rotated clockwise along its viewing axis.",
    "counterclockwise": "Counterclockwise rotation – The camera rotated counterclockwise along its viewing axis.",

    "no movement": "No movement – The camera did not change its position or orientation.",
}

single_dof_map = {
    "phi": ["leftward", "rightward"],
    "theta": ["upward", "downward"],
    "psi": ["clockwise", "counterclockwise"],
    "tx": ["left", "right"],
    "ty": ["up", "down"],
    "tz": ["forward", "backward"],
}

obj_centered_map = {
    "translation": {
        "options": ["left", "right"],
        "task_prompt": {
            "first_round": "You are part of a multi-agent system. Your task is to determine the camera movement based on the spatial relationship between the source image and the target image.",
            "later_round": "sdfdsfdsf",
        },
    },
    
    "rotation": {
        "options": ["leftward", "rightward"],
        "task_prompt": {
            "first_round": "sdfdsfdsf",
            "later_round": "sdfdsfdsf",
        },
    },
}

total_options_map = {
    "single-dof-cls": single_dof_map,
    "obj-centered-cls": obj_centered_map,
}

def _generate_options_text(options: list):
    text = "\n".join(f"{idx}. {camera_pose_explanation[f'{option}']}" for idx, option in enumerate(options))
    option_map = {idx: option for idx, option in enumerate(options)}
    return text, option_map

class PromptGenerator:
    def __init__(self, config: Any):
        self.config = config # global config

    def _postprocess_for_options(self, options: list) -> list:
        is_trap = self.config.STRATEGY.IS_TRAP
        if is_trap:
            options.append("no movement")

        is_shuffle = self.config.STRATEGY.IS_SHUFFLE
        if is_shuffle:
            random.shuffle(options)
        return options

    ### task1: single-dof ###
    def spatial_reasoning_prompt_single_dof(self, **kwargs) -> str:
        metadata = kwargs.get("metadata", None)
        # TODO: change to task 1
        options = total_options_map[self.config.EXPERIMENT.TASK_NAME][self.config.EXPERIMENT.TASK_SPLIT]["options"]
        options = self._postprocess_for_options(options)

        answer_candidates, option_map = _generate_options_text(options)

        # TODO: change to task 1
        prompt = """Input:
        You are given a source image (the first image you see) and a target image (the second image you see). They are in the same scene but from source viewpoint and target viewpoint, respectively.

        Task:
        Suppose you are holding a camera, starting from source image viewpoint and moving to target image viewpoint. Your task is to determine the camera movement between the two images.

        Answer Candidates: 
        {answer_candidates}

        Response Format:
        Clearly explain your reasoning inside `<rsn></rsn>` tags, and then, provide your final decision inside `<ans></ans>` tags. 

        Response Format Example:
        <rsn>My reason is...</rsn>
        <ans>the option you choose here</ans>
        """.format(
            answer_candidates=answer_candidates,
        )
        return prompt, option_map

    ### Prompt for VLM-Only spatial reasoning ###
    def spatial_reasoning_prompt_obj_centered(self, **kwargs) -> str:
        metadata = kwargs.get("metadata", None)
        
        optioins = total_options_map[self.config.EXPERIMENT.TASK_NAME][self.config.EXPERIMENT.TASK_SPLIT]["options"]
        options = self._postprocess_for_options(options)

        answer_candidates, option_map = _generate_options_text(optioins)

        prompt = """Input:
        You are given a source image (the first image you see) and a target image (the second image you see). They are in the same scene but from source viewpoint and target viewpoint, respectively.

        Task:
        Suppose you are holding a camera, starting from source image viewpoint and moving to target image viewpoint. Your task is to determine the camera movement between the two images.

        Answer Candidates: 
        {answer_candidates}

        Response Format:
        Clearly explain your reasoning inside `<rsn></rsn>` tags, and then, provide your final decision inside `<ans></ans>` tags. 

        Response Format Example:
        <rsn>My reason is...</rsn>
        <ans>the option you choose here</ans>
        """.format(
            answer_candidates=answer_candidates,
        )
        return prompt, option_map

    ### Prompt for multi-agents reasoning ###
    def image_caption_prompt(self, **kwargs):
        metadata = kwargs.get("metadata", None)
        # prompt = """Input:
        # You are given a source image (the first image you see) and a target image (the second image you see). They are inthe  same scene but from source viewpoint and target viewpoint, respectively.

        # Task:
        # You are part of a multi-agent system. Your task is to understand the content of the images and provide a caption for each image. The caption should be concise and descriptive, capturing the key elements, the relative spatial relationships, and the occlusion of objects in the images. 

        # Response Format:
        # - Provide a caption for the source image inside `<src_caption></src_caption>` tags.
        # - Provide a caption for the target image inside `<tgt_caption></tgt_caption>` tags.
        # - Compare them and provide your comparison inside `<comparison></comparison>` tags.

        # Example Response Format:
        # <src_caption>Describe the source image here.</src_caption>
        # <tgt_caption>Describe the target image here.</tgt_caption>
        # <comparison>Compare the two images here.</comparison>
        # """
        optioins = total_options_map[self.config.TASK.NAME][self.config.TASK.SPLIT]["options"]

        is_trap = self.config.STRATEGY.IS_TRAP
        if is_trap:
            optioins.append("no movement")

        is_shuffle = self.config.STRATEGY.IS_SHUFFLE
        if is_shuffle:
            random.shuffle(optioins)

        answer_candidates, option_map = _generate_options_text(optioins)
        prompt = """Input:
        You are given a source image (the first image you see) and a target image (the second image you see). They are in the same scene but from source viewpoint and target viewpoint, respectively.

        Task:
        You are part of a multi-agent system. Your task is to understand the content of the images and infer the camera movement between the two images.

        Answer Candidates:
        {answer_candidates}

        Response Format:
        - If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags, and provide your final answer inside `<ans></ans>` tags. 

        Example Response Format:
        Able to judge the camera movement based on images:
        <rsn>My reason is...</rsn>
        <ans>the option you choose here</ans>
        """.format(
            answer_candidates=answer_candidates,
        )
        return prompt, option_map, answer_candidates

    def spatial_reasoning_prompt_ma(self, vlm_answer: str, **kwargs) -> str:
        metadata = kwargs.get("metadata", None)
        # optioins = total_options_map[self.config.TASK.NAME][self.config.TASK.SPLIT]["options"]

        # is_trap = self.config.STRATEGY.IS_TRAP
        # if is_trap:
        #     optioins.append("no movement")

        # is_shuffle = self.config.STRATEGY.IS_SHUFFLE
        # if is_shuffle:
        #     random.shuffle(optioins)

        # answer_candidates, option_map = _generate_options_text(optioins)

        option_map = kwargs.get('option_map', None)
        answer_candidates = kwargs.get('answer_candidates', None)

        # prompt = """Input: 
        # You are given a VLM's description on a source image and a target image, which is about the spatial relationship between the two images. The VLM's answer is as follows:
        # "{vlm_answer}"

        # Task:
        # You are part of a multi-agent system. Your task is to determine the camera movement based on the VLM's observations.

        # Answer Candidates:
        # {answer_candidates}

        # Response Format:
        # - If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags, and provide your final answer inside `<ans></ans>` tags. 
        # - If you need more information to make a decision, then ask your question to VLM inside `<ques></ques>` tags.

        # Example Response Format:
        # 1. Able to judge the camera movement based on the VLM's observations:
        # <rsn>My reason is...</rsn>
        # <ans>the option you choose here</ans>

        # 2. Need more information to make a decision:
        # <ques>Can you provide more details about...</ques>
        # """.format(
        #     vlm_answer=vlm_answer,
        #     answer_candidates=answer_candidates,
        # )

        prompt = """Input: 
        You are given a VLM's description on a source image and a target image, along with its justification on camera movement between image. The VLM's answer is as follows:
        "{vlm_answer}"

        Task:
        You are part of a multi-agent system. Your task is to determine if VLM's decision and its reasoning is reasonable.

        Answer Candidates:
        {answer_candidates}

        Response Format:
        - If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags, and provide your final answer inside `<ans></ans>` tags. 
        - If you need more information to make a decision, then ask your question to VLM inside `<ques></ques>` tags.

        Example Response Format:
        1. Able to judge the camera movement based on the VLM's observations:
        <rsn>My reason is...</rsn>
        <ans>the option you choose here</ans>

        2. Need more information to make a decision:
        <ques>Can you provide more details about...</ques>
        """.format(
            vlm_answer=vlm_answer,
            answer_candidates=answer_candidates,
        )
        return prompt, option_map