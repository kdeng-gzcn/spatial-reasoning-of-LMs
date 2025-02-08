from SpatialVLM.Prompter.PrompterTemplate import PromptTemplate

class TaskDesc_Prompter4Pair(PromptTemplate):

    def __init__(self, **kwargs):

        self.direction = kwargs.get("direction", None)

        super().__init__()

    def __call__(self):

        prompt = "You are tasked with spatial reasoning evaluation, i.e. infering the camera movement from a source image to a target image. Although you can't see the pair of images directly, you have a friend Vision Language Model (VLM) who can answer all the questions that you ask about the images. My suggestions are: 1. to find out the main objects occur in both images; 2. find out the positions of the objects inside the image, e.g. there is a box on the left side of source image, and the same box is on the right side of target image; 3. judge the movement of camera from source image to target image based on information you get. For more information: 1. you can have several turns of conversation with VLM if you like; 2. the task is not hard because the camera movement is only by one of six direction (three translations and three rotations), and here you just need to judge if the camera is rotating leftward or rightward from source to target. Now suppose you are chatting with your friend, formulate questions for your friend. "

        return prompt

class LLM_To_VLM_Prompter4Pair(PromptTemplate):

    def __init__(self):

        super().__init__()

    def __call__(self, LLM_Questions):

        prompt = f"""You are a Vision Language Model (VLM) that can answer questions about images. Your friend, the Language Model (LLM), is trying to infer the camera movement from a source image to a target image. (Note that the first image you see is source image and the second one is target image) He said, "{LLM_Questions}" Please provide detailed answers to these questions based on the images you can see. Your answers should help the LLM determine whether the camera is rotating leftward or rightward from the source image to the target image. Just answer his questions, do not tell him your judgement. """

        return prompt

class VLM_To_LLM_Prompter4Pair(PromptTemplate):

    def __init__(self):

        super().__init__()
    
    def __call__(self, VLM_Answers):

        prompt = f"""Your friend, the Vision Language Model (VLM), has provided the following information about the images. He said, "{VLM_Answers}" Based on this, please choose one of the following options to determine the main camera movement from the source image to the target image: (0) Not confident enough to judge; (1) Leftward rotation; (2) Rightward rotation. Provide your answer inside the special token <ans></ans>, and explain your reasoning inside the tokens <rsn></rsn>. You may only answer with (1) or (2) if you are very confident. If you choose option (0), you may ask additional questions to VLM inside the special tokens <ques></ques>. Remember: 1. The camera movement is limited to a leftward or rightward rotation; 2. Your judgment should be based on the positions and movements of objects in the images as described by the VLM; 3. If the information provided by the VLM is insufficient, you can ask for more details by selecting option (0). """

        return prompt

if __name__ == "__main__":

    pass
