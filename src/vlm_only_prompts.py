task_prompt_individual_vlm_zero_shot = """Task: You are given two images. The source image (the first image you see) represents the initial camera view, while the target image (the second image you see) represents the view after potential movement. The camera movement is considered to have six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). Your task is to determine the significant camera movement from the source image to the target image.

Possible Answers:
- (0) {opt1}
- (1) {opt2}
- (2) {opt3}
- (3) {opt4}

Response Format:
- Explain your reasoning inside <rsn></rsn>.
- Provide your answer inside the special tokens <ans></ans>.

Example Response:
<rsn>My reason is...</rsn>
<ans>0</ans>
"""

addtional_info_zero_shot = """
Additional Context:
- The dataset consists of pairs of images, each pair showing the same scene from slightly different perspectives due to the camera movement. It is essential to identity the slight changes in object positions in the target image compared to the source image.
- {additional_info} This constraint simplifies the problem: there is no combination of multiple movements or complex transformations.
"""

CoT_zero_shot = """
Here are some reasoning skills:
Step1: You need to carefully identify the main objects that occurs in both images. 
Step2: You need to identify how the objects have moved in the target image compared to the source image.
{CoT_reasoning_skills}

Let's think step by step.
"""

VoT_zero_shot = """
Here are some reasoning skills:
You need to carefully identify the main objects (represented by ⭕️) that occurs in both images. Then, you can determine the direction of the camera movement based on how the objects have moved in the target image compared to the source image.
{VoT_reasoning_skills}

Visualize each reasoning step.
"""

CoT_prompt = """

"""

VoT_promopt = """

"""
