task_prompt = """You are given two images:  
- The **source image (the first image you see)** is captured from a first-person view, as if you're holding a camera and taking a photo while approaching an object.  
- The **target image (the second image you see)** shows the same scene after you've walked around and reoriented the camera to face the central object directly.  

Your goal is to infer the **major camera translation** (i.e., the movement of the camera in space) that occurred between the two views.  

Input:  
- Source Image and Target Image: Both depict roughly the same scene but from different viewpoints.

Your Task:  
Based on visible changes in perspective, alignment of the central object, and spatial arrangement of background elements, determine the most likely camera translation.

Possible Answers:
- (0) {opt1}
- (1) {opt2}
- (2) {opt3}
- (3) {opt4}

Response Format:
- Use <rsn></rsn> to explain your reasoning, focusing on changes in scene elements, alignment, and orientation.
- Use <ans></ans> to indicate your final choice (just the option number).

Example Response:
<rsn>We are in the left-side of the object in the source image, and we turn to the center of the object in the target image, indicating a right camera translation.</rsn>
<ans>0</ans>
"""

task_prompt_ablation = """You are given two images:  
- The **source image (the first image you see)** is captured from a first-person view, as if you're holding a camera and taking a photo while approaching an object.  
- The **target image (the second image you see)** shows the same scene after you've walked around and reoriented the camera to face the central object directly.  

Your goal is to infer the **major camera translation** (i.e., the movement of the camera in space) that occurred between the two views.  

Your Task:  
Based on changes from the following description in perspective, alignment of the central object, and spatial arrangement of background elements, determine the most likely camera translation.
{description}

Possible Answers:
- (0) {opt1}
- (1) {opt2}
- (2) {opt3}
- (3) {opt4}

Response Format:
- Use <rsn></rsn> to explain your reasoning, focusing on changes in scene elements, alignment, and orientation.
- Use <ans></ans> to indicate your final choice (just the option number).

Example Response:
<rsn>We are in the left-side of the object in the source image, and we turn to the center of the object in the target image, indicating a right camera translation.</rsn>
<ans>0</ans>
"""

short_answer_dict = {
    0: "unable to judge",
    1: "left",
    2: "right",
    3: "no movement"
}

detailed_answer_dict = {
    "left": "Left translation – The camera moved to the left horizontally.",
    "right": "Right translation – The camera moved to the right horizontally.",
    "no movement": "No movement – The two images are completely identical with no even slight changes.",
    "unable to judge": "Unable to judge – This option should only be selected in cases where the images are severely corrupted or lack sufficient visual information."
}