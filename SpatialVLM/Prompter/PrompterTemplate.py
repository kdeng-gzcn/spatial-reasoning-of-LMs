"""

No use, only for template and archive

"""

class PromptTemplate():
    
    def __init__(self, **kwargs):

        pass

    def __call__(self):

        raise NotImplementedError()

# class StartPrompt(PromptTemplate):

#     def __init__(self):

#         super().__init__()

#     def __call__(self, mode="pair"):

#         """

#         This prompt is for task description

#         """

#         if mode == "single":

#             prompt = r"You are tasked with understanding two viewpoints, which are source viewpoint and target viewpoint, of the same scene taken from different angles. Although you can't see these views directly, you have a friend within the scene who can provide detailed descriptions, including distances and angles related to camera. Now suppose you are chatting with your friend, formulate questions inside special token <question> <\question> for your friend about source viewpoint first to help you better understand source image."

#         if mode == "pair":

#             prompt = r"You are tasked with understanding the relative positioin between two viewpoints from camera, which are source image and target image, of the same scene taken from 2 different viewpoints. Although you can't see the 2 images directly, you have a friend VLM who can provide detailed descriptions of the images to you. My suggestion is to find out the main objects in each image first, and then find out the relative position of the main objects in further conversation, to judge the movement of camera from source image to target image. Note that, the task is not hard because you just need to judge if the camera is rotating leftward or rightward from source to target. Now suppose you are chatting with your friend, formulate questions for your friend. "

#         return prompt

# class EyePrompt(PromptTemplate):
#     """
#     Not use
#     """
#     def __init__(self):
#         super().__init__()

#     def __call__(self, questions=None):
#         """
#         Test
#         """

#         prompt = "Describe where the most prominent feature or object in the scene, also mention the direction of the object towards related to the camera."

#         return prompt

# class BrainPrompt(PromptTemplate):

#     def __init__(self):

#         super().__init__()
    
#     def answer_prompter_single(self, answer_from_VLM, source=True):

#         if source:
#             prompt = rf'The friend who has the source viewpoint said, """{answer_from_VLM}""". Now, you can ask the friend for viewpoint in target image based on what you see in source image.'
#         else:
#             prompt = rf'The friend who has the target viewpoint said, """{answer_from_VLM}""". Are you satisfied with these descriptions? If so, specify the direction you should move from the source to reach the target viewpoint. If not, continue asking relevant questions to obtain more details.'

#         return prompt
    
#     def answer_prompter_pair(self, answer_from_VLM):

#         """

#         Combinate source and target answer for new prompt for brain.

#         """

#         prompt = rf'The friend said, """{answer_from_VLM}""". Please choose one among the following options as your judgement for the main movement of camera from source to target: (0) not enough confident to judge (1) leftward (2) rightward. And please give your explanation for your answer inside special token <ans> <\ans> (if you choose (0), then give your further questions inside <ques> <\ques>)'

#         return prompt

if __name__ == "__main__":

    pass
