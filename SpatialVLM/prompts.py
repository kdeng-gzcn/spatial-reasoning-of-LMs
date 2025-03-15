task_prompt_individual_vlm_zero_shot = """Task: You are given two images. The source image (the first image you see) represents the initial camera view, while the target image (the second image you see) represents the view after potential movement. Your task is to determine the significant camera movement from the source image to the target image.

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
- The dataset only contains camera leftward or rightward rotations of around 15 degree.
"""

CoT_zero_shot = """
Here are some reasoning skills:
Step1: You need to carefully identify the main objects that occurs in both images. 
Step2: you need to determine the direction of the camera movement based on the relative position of the object in each image.
1. The camera rotated leftward horizontally.
if the main objects that you chose in the source image is on the left/middle/right side of the frame and the target image shows the object comes to the middle/right/more right side of the frame, the camera has rotated left.

2. The camera rotated rightward horizontally.
if the main objects that you chose in the source image is on the left/middle/right side of the frame and the target image shows the object comes to the left/middle/more left side of the frame, the camera has rotated right.
"""

VoT_zero_shot = """
Here are some reasoning skills:
You need to carefully identify the main objects (represented by ⭕️) that occurs in both images. Then, you can determine the direction of the camera movement based on the relative position of the object in each image.
1. The camera rotated leftward horizontally.
if the images look like:
--------         --------
|  ⭕️   | ---->  |    ⭕️ |
--------         --------
(source)         (target)

which means the camera has the movement:
⭕️    leftward     ⭕️
|    --------->   \ 
📷                 📷

2. The camera rotated rightward horizontally.
if the images look like:
--------         --------
|    ⭕️ | ---->  | ⭕️    |
--------         --------
(source)         (target)

which means the camera has the movement:
⭕️    rightward    ⭕️
|    --------->     /
📷                 📷
"""
