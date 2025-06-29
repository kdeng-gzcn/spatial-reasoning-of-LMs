task_prompt = """Task: You are given two images. The first image is the source view, as if you are holding a camera and taking a photo. The second image is the target view, captured after you walk around and rotate the camera.

Your goal is to infer the significant **yaw camera rotation** that occurred from the first to the second view.

Input:
- Source Image and Target Image: Both show roughly the same scene from different viewing angles.

Possible Answers:
- (0) {opt1}
- (1) {opt2}
- (2) {opt3}
- (3) {opt4}

Response Format:
- Use <rsn></rsn> to explain your reasoning, focusing on changes in scene elements, alignment, and orientation.
- Use <ans></ans> to indicate your final choice (just the option number).

Example Response:
<rsn>The window moved from the center to the right edge, which suggests the camera turned left.</rsn>
<ans>0</ans>
"""

short_answer_dict = {
    0: "unable to judge",
    1: "leftward",
    2: "rightward",
    3: "no movement"
}

detailed_answer_dict = {
    "leftward": "Leftward rotation – The camera rotated leftward horizontally.",
    "rightward": "Rightward rotation – The camera rotated rightward horizontally.",
    "no movement": "No movement – The two images are completely identical with no even slight changes.",
    "unable to judge": "Unable to judge – This option should only be selected in cases where the images are severely corrupted or lack sufficient visual information."
}