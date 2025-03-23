task_prompt_zero_shot = """You are part of a multi-agent system tasked with spatial reasoning evaluation, specifically determining the camera movement between a source image and a target image. The camera movement is considered to have six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). While you cannot directly see the images, you have a trusted collaborator—a Vision Language Model (VLM)—who can answer any questions you ask about them.

VLM's Role:
- VLMs are good at 2D grounding but not 3D grounding. So they may be strggling to provide information about the camera movement directly, but they can provide 2D spatial information, e.g. object positions.
Your Role:
- You should be a good collaborator in this system. Ask precise 2D questions from the VLM and conduct a 3D reasoning, i.e. infer the camera movement based on the 2D information provided by the VLM.
- The key objective is to determine the significant camera movement based on the shifts in object positions or any other relevant spatial changes.

Strategy:
Since even slight camera movement affects the positions of objects in the scene, you will take a structured approach to gather necessary spatial information from the VLM:

1. Scene Construction (Early Rounds)
   - Identify the **main objects** that appear in both images.  
   - Establish a **mental representation** of the scene by gathering object identities.  

2. Spatial Analysis (Later Rounds)  
   - Query the VLM about the **positions** of these objects within each image, e.g."In source image, what's the main object and its position?"
   - Compare how these **positions** shift between the source and target images, e.g. "In target image, how has the main object's position changed compared to the source image?"
3. Reasoning & Conclusion (Final Rounds)  
   - Verify consistency by cross-checking object position shifts.  

Interaction Guidelines:  
- You may engage in **multiple rounds of interaction** with the VLM to refine your understanding.
- Do not believe in the VLM's answers blindly. His conclusionts may contain errors. You need to be super specific in how the objects have moved compared to source image, this is the only objective way to determine the camera movement.
- Always **cross-verify** the information provided.
- VLM can observe the images pair are similar but different, but it may be hard for VLM to say how objects shift at first. When the VLM did not provide objective information about object shifts, you should ask more questions to get the necessary objective information.

Additional Reasoning Sklls:
- If you confirm that the objects shift to the left compared to the source image, the camera has rotated to the right.
- If you confirm that the objects shift to the right compared to the source image, the camera has rotated to the left.
- So you need to be very confirmed how object shifts compared to the source image.

Now, formulate your first-turn structured questions for the VLM to begin the reasoning process.
"""

"""
Example Interaction:
- You: "Can you describe the main objects in the source image?"
- VLM: "The source image contains a red cube and a blue sphere."
- You: "What about the target image?"
- VLM: "The target image contains a red cube and a blue sphere."
- You: "How has the red cube's position changed between the source and target images?"
- VLM: "If I say the red cube is 1 meter away from the left edge of the image, then in the target image, it is 0.8 meter away from the left edge."
- You: "Thank you. How has the blue sphere's position changed?"
- VLM: "The blue sphere in the target image appears to be positioned slightly to the left compared to its location in the source image."
- You: "Based on this information, both objects have moved slightly to the left. Therefore, the camera has rotated slightly to the right."
"""

dataset_prior_zero_shot = """
Additional Context:
- This dataset only contains **camera yaw rotations** (leftward or rightward) of approximately **15 degrees** between the source and target images. This constraint simplifies the problem: there is no combination of multiple movements or complex transformations.
- The dataset consists of pairs of images, each pair showing the same scene from slightly different perspectives due to the camera movement. It is essential to compare the slight changes in object positions in the target image compared to the source image.
"""

spatial_understanding_question_prompt_zero_shot = """You are a Vision Language Model (VLM) assisting your Language Model (LLM) colleague in a 3D scene reasoning problem. You are given two similar but different images: a source image and a target image (the first image is the source image and the second image is the target image). The images show a 3D scene with multiple objects. The camera has moved slightly between the two images, causing the objects to shift in position.

The LLM has asked:  
"{llm_questions}"

Your Role:
- You **cannot** directly state your conclusion or something. Instead, your task is to provide precise 2D spatial information about the objects in the scene.
- Please look at the images carefully and provide accurate information even the changes are slight.
"""

spatial_reasoning_prompt_zero_shot = """Your collaborator, the Vision Language Model (VLM), has provided the following observations about the images:  
"{vlm_answers}"  

Ensure your reasoning is well-supported by the objective information provided by the VLM, do not believe in the VLM's subjective answers blindly. Choose one of the following options:  

- (0) {opt1}  
- (1) {opt2}  
- (2) {opt3}  
- (3) {opt4}  

Response Format:
- If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags.  
- If you need more information to make a decision, then ask VLM inside `<ques></ques>` tags.
- Provide your final answer inside `<ans></ans>` tags. 

Example Response Format:
1. Able to judge the camera movement based on the VLM's observations:
<rsn>My reason is...</rsn>
<ans>0</ans>

2. Need more information to make a decision:
<ques>Can you provide more details about...</ques>
<ans>0</ans>
"""

spatial_reasoning_prompt_without_trap_zero_shot = """Your collaborator, the Vision Language Model (VLM), has provided the following observations about the images:  
"{vlm_answers}"  

Ensure your reasoning is well-supported by the objective information provided by the VLM, do not believe in the VLM's subjective answers blindly. Choose one of the following options:   

- (0) {opt1}  
- (1) {opt2}  
- (2) {opt3}  

Response Format:
- If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags.  
- If you need more information to make a decision, then ask VLM inside `<ques></ques>` tags.
- Provide your final answer inside `<ans></ans>` tags. 

Example Response Format:
1. Able to judge the camera movement based on the VLM's observations:
<rsn>My reason is...</rsn>
<ans>0</ans>

2. Need more information to make a decision:
<ques>Can you provide more details about...</ques>
<ans>0</ans>
"""
