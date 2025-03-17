task_prompt_zero_shot = """You are part of a multi-agent system tasked with spatial reasoning evaluation, specifically inferring the camera movement from a source image to a target image. While you cannot directly see the images, you have a trusted collaborator—a Vision Language Model (VLM)—who can answer any questions you ask about them.  

To systematically deduce the camera movement, follow these steps:  
1. Identify the main objects present in both images.  
2. Determine the positions of these objects within each image.  
3. Analyze changes in object placement to infer the camera movement.  

Guidelines:  
- You may have multiple rounds of interaction with the VLM to refine your understanding.   

Now, formulate precise and structured questions for your VLM collaborator to gather the necessary spatial information.
"""

dataset_prior_zero_shot = """
The camera movement is restricted to one of six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). 
"""

spatial_understanding_question_prompt_zero_shot = """You are a Vision Language Model (VLM) assisting your Language Model (LLM) colleague in inferring the camera movement between a source image and a target image. (The first image you see is the source image, and the second one is the target image.)  

The LLM has asked:  
"{llm_questions}"  

Please analyze both images carefully and provide detailed answers, focusing on:  
- The primary objects appearing in both images.  
- The positions of these objects within each image.  
- Any noticeable changes in their locations, orientations, or sizes between the source and target images.  

Your response should strictly answer the LLM's questions without directly stating the inferred camera movement.
"""

spatial_reasoning_prompt_zero = """Your collaborator, the Vision Language Model (VLM), has provided the following observations about the images:  
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
