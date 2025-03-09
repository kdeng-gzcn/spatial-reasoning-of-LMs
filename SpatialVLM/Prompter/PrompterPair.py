import numpy as np

from SpatialVLM.Prompter.PrompterTemplate import PromptTemplate

class TaskDesc_Prompter4Pair(PromptTemplate):

    def __init__(self, **kwargs):

        self.if_give_example = kwargs.get("if_give_example", False)

        super().__init__()

    def __call__(self):

        prompt = "You are tasked with spatial reasoning evaluation, i.e. infering the camera movement from a source image to a target image. Although you can't see the pair of images directly, you have a friend Vision Language Model (VLM) who can answer all the questions that you ask about the images. My suggestions are: 1. to find out the main objects occur in both images; 2. find out the positions of the objects inside the image; 3. judge the movement of camera from source image to target image based on information you get. For more information: 1. you can have several turns of conversation with VLM if you like; 2. the task is not hard because the camera movement is limited to only be one of six direction (three translations and three rotations), e.g. only left or right translation/translation. Now suppose you are chatting with your friend, formulate questions for your friend. "

        self.if_give_example = True

        if self.if_give_example:
            prompt += """Example: LLM: "Are there any objects that appear in both images?", VLM: "Yes, there is a chair and a table in both images.", LLM: "Where is the chair in the source image?"
            VLM: "The chair is on the left side of the image.", LLM: "Where is the chair in the target image?", VLM: "The chair is now on the right side of the image.", LLM: "Given this information, it seems that the camera has rotated to the left, since the entire scene shifts rightward in the image plane, making objects appear to move from left to right. If the camera had no movement or rotated rightward, the chair would have either remained on the left side of the image or moved further left." """

        return prompt

class LLM_To_VLM_Prompter4Pair(PromptTemplate):

    def __init__(self, **kwargs):

        self.if_give_example = kwargs.get("if_give_example", False)

        super().__init__()

    def __call__(self, LLM_Questions):

        prompt = f"""You are a Vision Language Model (VLM) that can answer questions about images. Your friend, the Language Model (LLM), is trying to infer the camera movement from a source image to a target image. (Note that the first image you see is source image and the second one is target image) He said, "{LLM_Questions}" Please provide detailed answers to these questions based on the images you see, your responses should focus on describing the positions, movements, and relationships of objects in the images. Just answer his questions, do not directly tell him your judgement. Note that the camera movement is limited to only be one of six direction (three translations and three rotations), e.g. only left or right translation/translation. """

        self.if_give_example = False

        if self.if_give_example:
            prompt += """Example: LLM: "Are there any objects that appear in both images?", VLM: "Yes, there is a chair and a table in both images.", LLM: "Where is the chair in the source image?"
            VLM: "The chair is on the left side of the image.", LLM: "Where is the chair in the target image?", VLM: "The chair is now on the right side of the image.", LLM: "Given this information, it seems that the camera has rotated to the left, since the entire scene shifts rightward in the image plane, making objects appear to move from left to right. If the camera had no movement or rotated rightward, the chair would have either remained on the left side of the image or moved further left." """

        return prompt

class VLM_To_LLM_Prompter4Pair(PromptTemplate):

    def __init__(self):

        super().__init__()
    
    def __call__(self, VLM_Answers):

        option_map = {
            0: "ask more questions",
            1: "leftward rotation",
            2: "rightward rotation",
            3: "no movement",
        }

        option_values = list(option_map.values())
        np.random.shuffle(option_values)
        self.option_map = {i: option_values[i] for i in range(len(option_values))}

        # prompt = f"""Your friend, the Vision Language Model (VLM), has provided the following information about the images. He said, "{VLM_Answers}" Based on this, please choose one of the following options to determine the main camera movement from the source image to the target image: (0) {self.option_map[0]}; (1) {self.option_map[1]}; (2) {self.option_map[2]}; (3) {self.option_map[3]}. Provide your answer inside the special tokens <ans></ans>, e.g. <ans>0</ans>, and explain your reasoning inside the special tokens <rsn></rsn>, e.g. <rsn>My reason is...</rsn>. If you choose the option that asks more questions, you may ask additional questions to VLM inside the special tokens <ques></ques>, e.g. <ques>My questions are...</ques>. Note that: 1. The actual camera movement is limited to a leftward or rightward rotation; 2. Your judgment should be based on the positions and movements of objects in the images as described by the VLM. """

        prompt = f"""Your friend, the Vision Language Model (VLM), has provided the following information about the images. He said, "{VLM_Answers}" Based on this, please choose one of the following options to determine the main camera movement from the source image to the target image: (0) {self.option_map[0]}; (1) {self.option_map[1]}; (2) {self.option_map[2]}; (3) {self.option_map[3]}. Provide your answer inside the special tokens <ans></ans>, e.g. <ans>0</ans>, and explain your reasoning inside the special tokens <rsn></rsn>, e.g. <rsn>My reason is...</rsn>. If you choose the option that asks more questions, you may ask additional questions to VLM inside the special tokens <ques></ques>, e.g. <ques>My questions are...</ques>. Note that your judgment should be based on the positions and movements of objects in the images as described by the VLM. """

        self.option_map = {k: v.replace("leftward rotation", "leftward").replace("rightward rotation", "rightward") for k, v in self.option_map.items()}

        return prompt

if __name__ == "__main__":

    pass
