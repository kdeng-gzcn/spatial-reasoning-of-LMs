from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support."},
    {"role": "user", "content": "What day is today?"},
    {"role": "assitant", "content": "Today is 6th Nov."},
    {"role": "user", "content": "Can you repeat the previous question I ask you and the answer you said to me?"},
]

# 1. map raw messages to model-form ids and attention mask 
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(model.device)

# print(inputs)

# 2. **dict to get ids and attention mask
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 3. decode ids and print the model-form messages
# print("decode string", tokenizer.decode(input_ids[0], skip_special_tokens=True))

# 4. setting terminators in model
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 5. run model inference
# MAIN generate function
input_output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# print("decode string",tokenizer.decode(input_output_ids[0]))

# 6. process, squeeze output ids and remoce input ids
output_ids = input_output_ids[0][input_ids.shape[-1]:]

# print(output_ids)

# visual answers: from numbers to text
print("decode processed string1 from llama:", tokenizer.decode(output_ids, skip_special_tokens=False))

# try history
messages.append({"role": "assitant", "content": tokenizer.decode(output_ids, skip_special_tokens=True)})
messages.append({"role": "user", "content": "What day is tomorrow?"})

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(model.device)

input_output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

input_ids = inputs['input_ids']

output_ids = input_output_ids[0][input_ids.shape[-1]:]

print("decode processed string2 from llama:", tokenizer.decode(output_ids, skip_special_tokens=False))

# Done!
print("FULL CONVERSATION: ", tokenizer.decode(input_output_ids[0], skip_special_tokens=False))
