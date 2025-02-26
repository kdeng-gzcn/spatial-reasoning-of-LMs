import numpy as np

from SpatialVLM.Prompter.PrompterTemplate import PromptTemplate

class TaskDesc_Prompter4Baseline(PromptTemplate):

    def __init__(self, **kwargs):

        self.direction = kwargs.get("direction", None)

        super().__init__()

    def __call__(self):

        option_map = {
            0: "unable to judge",
            1: "leftward",
            2: "rightward",
            3: "no movement",
        }

        option_values = list(option_map.values())
        np.random.shuffle(option_values)
        self.option_map = {i: option_values[i] for i in range(len(option_values))}

        prompt = f"You are tasked with spatial reasoning evaluation, i.e. infering the camera movement from a source image to a target image (Note that the image1 you see is source image and the image2 is target image). Note that, in this task, the camera movement is limited to be only leftward rotationrightward rotation. Please give your answer from the following options to determine the main camera movement from the source image to the target image: (0) {self.option_map[0]}; (1) {self.option_map[1]}; (2) {self.option_map[2]}; (3) {self.option_map[3]}. Provide your answer inside the special tokens <ans></ans>, e.g. <ans>0</ans>, and explain your reasoning inside the special tokens <rsn></rsn>. "

        return prompt