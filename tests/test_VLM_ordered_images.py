import sys
sys.path.append("./")

import requests
from PIL import Image
from io import BytesIO

cat_url = "https://www.washingtonian.com/wp-content/uploads/2018/02/CatContest2-1024x683.jpg"
cat_response = requests.get(cat_url)
cat_image = Image.open(BytesIO(cat_response.content))

dog_url = "https://placedog.net/800/600"
dog_response = requests.get(dog_url)
dog_image = Image.open(BytesIO(dog_response.content))

from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 

images = [cat_image, dog_image]
placeholder = "<|image_1|>\n<|image_2|>\n"

messages = [
    {
        "role": "user", 
        "content": placeholder+"(Note that the first image you see is source image and the second one is target image)Please describe the main object in source image and in target image."
    },
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print()
print(response)
