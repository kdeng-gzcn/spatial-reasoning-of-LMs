short_dict = {
    0: "unable to judge",
    1: "leftward",
    2: "rightward",
    3: "no movement",
}

detailed_dict = {
    "leftward": "Leftward rotation – The camera rotated leftward horizontally.",
    "no movement": "No movement – The two images are completely identical with no even slight changes. This option should only be selected if there is absolute certainty that no movement at all as occurred.",
    "unable to judge": "Unable to judge – This option should only be selected in cases where the images are severely corrupted, fail to load, or lack sufficient visual information to determine movement.",
    "rightward": "Rightward rotation – The camera rotated rightward horizontally.",
}

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
- This dataset only contains **camera yaw rotations (leftward or rightward)**  of approximately **15 degrees** between the source and target images. This constraint simplifies the problem: there is no combination of multiple movements or complex transformations.
- The dataset consists of pairs of images, each pair showing the same scene from slightly different perspectives due to the camera movement. It is essential to identity the slight changes in object positions in the target image compared to the source image.
"""

"""
- Case1: The camera rotated leftward horizontally:
    if the main objects that you chose in the source image is on the left/middle/right side of the frame and the target image shows the object comes to the middle/right/more right side of the frame, then the camera has rotated left.

- Case2: The camera rotated rightward horizontally:
    if the main objects that you chose in the source image is on the left/middle/right side of the frame and the target image shows the object comes to the left/middle/more left side of the frame, the camera has rotated right.
"""

"""
- Case1: The camera rotated leftward horizontally:
    if the images look like:
    --------         --------
    |  ⭕️   | ---->  |    ⭕️ |
    --------         --------
    (source)         (target)

    which means the camera has the movement:
    ⭕️    leftward     ⭕️
    |    --------->   \ 
    📷                 📷

- Case2: The camera rotated rightward horizontally:
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