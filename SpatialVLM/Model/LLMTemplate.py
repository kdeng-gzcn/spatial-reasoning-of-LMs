from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

class LLMTemplate:

    def __init__(self):
        pass
        
    def __call__(self):
        pass
        
class HuggingFaceLLM(LLMTemplate):

    def __init__(self, name=None):
        super().__init__()
        
        self.model_name = name
        self.messages = [
            {"role": "system", "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support."},
        ]

    def clearhistory(self):
        
        self.messages = [
            {"role": "system", "content": "You are a warm and friendly chatbot who is always eager to help and offer kind words of support."},
        ]
        
    def __call__(self):
        """
        load model and tokenizer from huggingface
        """

        model_id = self.model_name

        # 1. load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # 2bytes
            device_map="auto",
        )

        self.model = model
        self.tokenizer = tokenizer
        
    def pipeline(self, prompt: str = None):
        
        # 0. add usr content
        self.messages.append({"role": "user", "content": prompt})

        # 1. chat template for specific model + tokenizer    
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        input_ids = inputs['input_ids']
        # 1.1* for llama, you need to set up terminators
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 2. run model inference and store full conversation
        input_output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id, # avoid warning
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        self.full_history = self.tokenizer.decode(input_output_ids[0], skip_special_tokens=False)
        # 2.1 decode manually
        output_ids = input_output_ids[0][input_ids.shape[-1]:]
        # 2.2 decode by tokenizer
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 3. add chat history
        self.messages.append({"role": "assitant", "content": answer})
        
        return answer
    
if __name__ == "__main__":
    
    model = HuggingFaceLLM(name="meta-llama/Meta-Llama-3-8B-Instruct")
    # load cache
    model()
    output = model.pipeline("Who are you?")
    print(output)
