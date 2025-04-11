archive_task_prompt_zero_shot = """You are part of a multi-agent system tasked with spatial reasoning evaluation, specifically determining the camera movement between a source image and a target image. The camera movement is considered to have six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). While you cannot directly see the images, you have a trusted collaborator—a Vision Language Model (VLM)—who can answer any questions you ask about them.

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

Now, formulate your first-turn structured questions for the VLM to begin the reasoning process.
"""

task_prompt_zero_shot = """You are part of a multi-agent system tasked with spatial reasoning evaluation, specifically determining the camera movement between a source image and a target image. The camera movement is considered to have six possible directions: three translations (left/right, up/down, forward/backward) and three rotations (around different axes). While you cannot directly see the images, you have a trusted collaborator—a Vision Language Model (VLM)—who can answer any questions you ask about them.

**VLM's Role:**
- VLMs excel at providing 2D spatial information, such as object positions and relative relationships between objects. However, VLMs are not equipped for 3D grounding, and they may struggle to directly identify the camera movement.
- The VLM can help answer questions regarding object positions, orientations, and any visible changes between images.

**Your Role:**
- As the reasoning agent, your role is to ask specific, structured 2D questions to the VLM and infer the 3D camera movement based on the 2D data provided.
- The goal is to determine the camera's movement by examining shifts in object positions and spatial relationships in the images.

**Strategy:**

1. **Scene Construction (Early Rounds):**
   - Identify the **main objects** present in both images (source and target).
   - Form a **mental representation** of the scene based on object identities.

2. **Spatial Analysis (Later Rounds):**
   - Ask the VLM detailed questions about the **positions** and **relative placement** of the main objects in each image. For example: "In the source image, what is the position of the main object relative to the background?"
   - Request information on how these **positions change** between the source and target images. For example: "In the target image, how has the main object's position changed relative to its position in the source image?"

3. **Reasoning & Conclusion (Final Rounds):**
   - Based on the position shifts provided by the VLM, reason about the **camera's movement**. Determine if the object shifts indicate translation or rotation, and infer the most likely camera movement direction.
   - **Cross-check** and **validate** consistency by asking the VLM to further clarify or revalidate object shifts. This ensures you arrive at an accurate conclusion about the camera movement.

**Interaction Guidelines:**
- **Multiple rounds of interaction** may be necessary to refine your understanding of the spatial changes between the images.
- Be **precise** in your questioning to gather **objective, measurable information** about how objects have shifted between the two images. This is critical for determining the camera movement.
- Always **verify** the answers you receive from the VLM. Be aware that its responses may not be entirely accurate, and it's important to validate the information through precise follow-up questions.
- If the VLM does not provide enough objective information on how the objects have shifted, **ask more focused questions** to gather the necessary data about object movement and spatial relationships.

Now, formulate your first-turn structured questions for the VLM to begin the reasoning process.
"""


dataset_prior_zero_shot = """
Additional Context:
- The dataset consists of pairs of images, each pair showing the same scene from slightly different perspectives due to the camera movement. It is essential to identity the slight changes in object positions in the target image compared to the source image.
- {additional_info} This constraint simplifies the problem: there is no combination of multiple movements or complex transformations.
"""


archive_spatial_understanding_question_prompt_zero_shot = """You are a Vision Language Model (VLM) assisting your Language Model (LLM) colleague in a 3D scene reasoning problem. You are given two similar but different images: a source image and a target image (the first image is the source image and the second image is the target image). The images show a 3D scene with multiple objects. The camera has moved slightly between the two images, causing the objects to shift in position.

The LLM has asked:  
"{llm_questions}"

Your Role:
- You **cannot** directly state your conclusion or something. Instead, your task is to provide precise 2D spatial information about the objects in the scene.
- Please look at the images carefully and provide accurate information even the changes are slight.
"""

spatial_understanding_question_prompt_zero_shot = """You are a Vision Language Model (VLM) assisting your Language Model (LLM) colleague in a 3D scene reasoning task. You are given two similar but different images: a source image and a target image. Both images show a 3D scene with multiple objects, and the camera has moved slightly between the two images, causing the objects to shift in position.

The LLM has asked:  
"{llm_questions}"

**Input:**
- You will be given a **pair of similar images**: first one is the source image, and the second one is the target image. The images are similar but differ slightly due to the camera movement between them.

**Your Role:**
- You **cannot** directly state conclusions or inferences about the camera movement or the 3D reasoning.
- Your task is to provide **precise 2D spatial information** about the objects in the scene based on the images you have. This could involve identifying:
  - Object positions (relative to the scene and other objects)
  - Spatial relationships between objects (e.g., which object is to the left or right of another)
  - Directional shifts in objects (e.g., does an object appear to have moved up, down, left, or right between the images?).
- Pay close attention to even the **slightest changes** in object positions between the two images and describe them as accurately as possible.

**What You Need to Do:**
1. **Accurately describe the position** of each object in both the source and target images. Describe any changes in the positions of the objects.
2. For each object, provide **clear 2D location details**, such as:
   - Where is the object located in the source image? (e.g., left, right, top, bottom, center of the frame)
   - Where is the object located in the target image? (e.g., has it shifted, and if so, how?)
3. Identify **relative movements** of the objects: Did any object move left/right, up/down, or closer/further from the camera between the source and target images?

**Important Notes:**
- **Focus on 2D spatial changes**. You should not infer the 3D camera movement directly, but instead, focus on the **shifts in positions** of the objects within the 2D images.
- Ensure your answers are **precise** and describe even minor shifts in the positions of objects.
- **Cross-check** between the source and target images for accuracy. If there is ambiguity or minor differences, be sure to highlight these in your response.
"""


spatial_reasoning_prompt_without_trap_zero_shot = """Your collaborator, the Vision Language Model (VLM), has provided the following observations about the images:  
"{vlm_answers}"  

Ensure your reasoning is well-supported by the objective information provided by the VLM, choose one of the following options:   

- (0) {opt1}  
- (1) {opt2}  
- (2) {opt3}  

Response Format:
- If you are confident to make decision, clearly explain your reasoning inside `<rsn></rsn>` tags.  
- If you need more information to make a decision, then ask VLM inside `<ques></ques>` tags.
- Provide your final answer inside `<ans></ans>` tags. 

Example Response Format:
1. If you are able to judge the camera movement based on the VLM's observations, then answer as this format:
<rsn>My reason is...</rsn>
<ans>0</ans>

2. If you think VLM provides misleading information or you need more conversations to make a decision, then answer as this format:
<ques>Can you provide more details about...</ques>
<ans>0</ans>
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
