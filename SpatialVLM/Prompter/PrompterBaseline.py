from SpatialVLM.Prompter.PrompterTemplate import PromptTemplate

class TaskDesc_Prompter4Baseline(PromptTemplate):

    def __init__(self, **kwargs):

        self.direction = kwargs.get("direction", None)

        super().__init__()

    def __call__(self):

        prompt = "You are tasked with spatial reasoning evaluation, i.e. infering the camera movement from a source image to a target image (Note that the image1 you see is source image and the image2 is target image). Please give your answer from the following options to determine the main camera movement from the source image to the target image: (0) Unable to judge; (1) Leftward rotation; (2) Rightward rotation. Provide your answer in text inside the special tokens <ans></ans>, e.g. <ans>leftward</ans>, <ans>rightward</ans>, and explain your reasoning inside the special tokens <rsn></rsn>. "

        return prompt