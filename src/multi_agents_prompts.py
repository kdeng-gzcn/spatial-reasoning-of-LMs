task_prompt_zero_shot = """You are part of a multi-agent system tasked with spatial reasoning evaluation, specifically determining the camera movement between a source image and a target image. While you cannot directly see the images, you have a trusted collaborator—a Vision Language Model (VLM)—who can answer any questions you ask about them.

Strategy:  
Since even slight camera movement affects the relative positions of objects in the scene, you will take a structured approach to gather necessary spatial information from the VLM:

1. Scene Construction (Early Rounds)  
   - Identify the **main objects** that appear in both images.  
   - Establish a **mental representation** of the scene by gathering object identities.  

2. Spatial Analysis (Later Rounds)  
   - Query the VLM about the **relative positions** of these objects within each image.  
   - Compare how these positions shift between the source and target images.  

3. Reasoning & Conclusion (Final Rounds)  
   - Verify consistency by cross-checking object position shifts.  

Interaction Guidelines:  
- You may engage in **multiple rounds of interaction** with the VLM to refine your understanding.  
- Questions should be **precise and structured** to efficiently extract useful spatial information.  

Now, formulate your first structured question for the VLM to begin the reasoning process.
"""

dataset_prior_zero_shot = """
Additional Context:
- The camera movement is considered to have six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). 
- This dataset only contains **camera yaw rotations** (leftward or rightward) of approximately **15 degrees**. This constraint simplifies the problem: instead of handling multiple types of transformations, the goal is solely to determine **whether the camera rotated left or right**. 
"""

spatial_understanding_question_prompt_zero_shot = """You are a Vision Language Model (VLM) assisting your Language Model (LLM) colleague in inferring the camera movement between a source image and a target image. (The first image you see is the source image, and the second one is the target image.)  

The LLM has asked:  
"{llm_questions}"

Your response should strictly answer the LLM's questions, without directly stating or inferring the camera movement. Provide detailed spatial information to help the LLM make an informed decision.
"""

spatial_reasoning_prompt_zero_shot = """Your collaborator, the Vision Language Model (VLM), has provided the following observations about the images:  
"{vlm_answers}"  

Based on this information, determine the primary camera movement from the source image to the target image. Ensure your reasoning is well-supported by the information provided by the VLM. Choose one of the following options:  

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
