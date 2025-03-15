import os
from typing import List, Any, Tuple
import base64
import io
from PIL import Image
import logging

import torch
from torchvision.transforms import ToPILImage

# load SpaceLLaVA
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# load general hf VLM
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
# load llava-Next
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# load paligemma
from transformers import PaliGemmaForConditionalGeneration
# load idefics2
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration

from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 0. Original Template For VLM
class VLMTemplate:

    def __init__(self, name: str = None):
        self.logger = logging.getLogger(__name__)

        self.Tensor2PIL = ToPILImage()
        self.model_name = name
        self.model = None
        self.processor = None
        self.message = None
    
    def pipeline(self, image=None, prompt: str = None):
        raise NotImplementedError


class HuggingFaceVLM(VLMTemplate):

    def __init__(self, name: str = None):
        super().__init__(name=name)

        # conversation
        self.message = None
        
    def __call__(self):
        
        # 1. parse model id
        model_id = self.model_name

        # 2. switch model (llava-Next, Paligemma, ide2-8b)
        if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":

            processor = LlavaNextProcessor.from_pretrained(model_id)
            model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")

        if model_id == "google/paligemma-3b-mix-224":

            device = "cuda:0"
            dtype = torch.bfloat16
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                revision="bfloat16",
            ).eval()
            processor = AutoProcessor.from_pretrained(model_id)
        
        # 3. return
        self.model = model
        self.processor = processor
        
    def pipeline(self, image=None, prompt: str = None):
        
        # 0. image pre-processing
        image = self.Tensor2PIL(image)

        # 1. switch model (differenct model has different pipeline template
        if 1:
            # 1. built chat history
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                    ],
                },
            ]
            
            # 2. apply chat template
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # 3. tokenizer the chat history
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

            # 4. run model inference
            output = self.model.generate(
                **inputs, 
                pad_token_id=self.processor.eos_token_id,
                max_new_tokens=200
                )
            
            # 5. decode output (with history)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        return answer

class LlavaNextVLM(VLMTemplate):

    def __init__(self, name=None):

        super().__init__(name=name)
        
    def load_model(self, model_id: str = None):
        
        # 1. parse model id
        self.model_name = model_id

        # 2. load model
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
            ).to("cuda:0")
        
        # 3. return
        self.model = model
        self.processor = processor
        
    def pipeline(self, image=None, prompt: str = None):
        
        # 0. image pre-processing
        image = self.Tensor2PIL(image)

        # 1. built chat history
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                ],
            },
        ]
        
        # 2. apply chat template (return str)
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # print(prompt)
        
        # 3. tokenizer the chat history (return dic with tensor and mask)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
        input_ids = inputs['input_ids']

        # 4. run model inference (return 2d tensor)
        output = self.model.generate(
            **inputs,
            max_new_tokens=200
            )

        # print(self.processor.decode(output[0], skip_special_tokens=True))
        
        # 5. decode output (with history)
        output_ids = output[0][input_ids.shape[-1]:] # [0] because its a 2d mat tensor, 
        answer = self.processor.decode(output_ids, skip_special_tokens=True)
        
        return answer

class Idefics2VLM(VLMTemplate):

    """
    
    This is for the pair of images
    
    """

    def __init__(self, name: str = None):
        super().__init__(name=name)
        
    def load_model(self, model_id: str = None):
        
        # 1. parse model id
        self.model_name = model_id

        # 2. load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = Idefics2Processor.from_pretrained(model_id)
        model = Idefics2ForConditionalGeneration.from_pretrained(model_id)
        model.to(self.device)
        
        # 3. return
        self.model = model
        self.processor = processor
        
    def pipeline(self, images=None, prompt: str = None):

        assert isinstance(images, list), "images should be a list!"
        
        # 0. image pre-processing (a pair of images)
        images = [self.Tensor2PIL(image) for image in images]

        # 1. built chat history
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                {"type": "image"},
            ],
        }]
        
        # 2. apply chat template (return str)
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # print(prompt)
        
        # 3. tokenizer the chat history (return dic with tensor and mask)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")
        input_ids = inputs['input_ids']

        # 4. run model inference (return 2d tensor) 2d because in batch case, we are using 1d case
        output = self.model.generate(
            **inputs,
            max_new_tokens=512
            )

        # print(self.processor.decode(output[0], skip_special_tokens=True))
        
        # 5. decode output (with history)
        output_ids = output[0][input_ids.shape[-1]:] # [0] because its a 2d mat tensor, 
        answer = self.processor.decode(output_ids, skip_special_tokens=True)
        # generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
        
        return answer

class Phi3VLM(VLMTemplate):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.conversation = []
        
    def _load_weight(self, model_id=None):
        
        # 1. parse model id
        if self.model_name == None:

            assert model_id is not None, "Need a model_id"
            
            self.model_name = model_id

        # 2. load model
        processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            num_crops=4
        ) 

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", # bfloat16
            _attn_implementation='flash_attention_2'    
        )
        
        # 3. return
        self.model = model
        self.processor = processor
        
    def pipeline(self, images=None, prompt: str = None):
        
        # 0. image pre-processing
        images = [self.Tensor2PIL(image) for image in images]

        prompt = "<|image_1|>\n<|image_2|>\n" + prompt

        # 1. built chat history
        conversation = [
            {
                "role": "user", 
                "content": prompt,
            },
        ]
        
        # 2. apply chat template (return str)
        prompt = self.processor.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # print(prompt)
        
        # 3. tokenizer the chat history (return dic with tensor and mask)
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 1024, 
            # "temperature": 0.0, 
            "do_sample": False,
        } 

        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        # print(self.processor.decode(generate_ids[0], skip_special_tokens=True))
        
        # 5. decode output (with history)
        answer = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0] 
        
        return answer
    
class Phi3VLMHistory(VLMTemplate):
    """
    histroy func can not be used in the pipeline
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self, model_id=None):
        processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            num_crops=4
        ) 

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", # bfloat16
            _attn_implementation='flash_attention_2'    
        )
        
        self.model = model
        self.processor = processor

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images=None, prompt: str = None):
        images = [self.Tensor2PIL(image) for image in images]
        prompt = "<|image_1|>\n<|image_2|>\n" + prompt
        self.conversation.append(
            {
                "role": "user", 
                "content": prompt,
            },
        )

        prompt = self.processor.tokenizer.apply_chat_template(
            self.conversation, 
            tokenize=False, 
            add_generation_prompt=True,
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 1024, 
            # "temperature": 0.0, 
            "do_sample": False,
        } 

        generate_ids = self.model.generate(
            **inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args,
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        
        answer = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0] 

        self.conversation.append(
            {
                "role": "assitant", 
                "content": answer,
            },
        )
        
        return answer


class SpaceLLaVA(VLMTemplate):

    def __init__(self, path="/bask/projects/j/jlxi8926-auto-sum/kdeng/doppelgangers/data/doppelgangers_dataset/doppelgangers/"):
        super().__init__()

        self.model_name = "remyxai/SpaceLLaVA"
        
        self.model_path = path
        
    def __call__(self):
        """
        This maybe a __call__
        """
        drive_path = self.model_path 
    
        # load pretrained weights
        mmproj = os.path.join(drive_path, "mmproj-model-f16.gguf")
        model_path = os.path.join(drive_path, "ggml-model-q4_0.gguf")
        
        # load model
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=False)
        spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers=-1, verbose=False)
        
        self.model = spacellava

    # for 2 imgs in the same prompt
    def image_to_base64_data_uri(self, image_inputs):
        """
        This function accepts a single image path, a single PIL Image instance, or a list of them.
        It returns the Base64-encoded data URI(s) for the image(s).
        """
        # If the input is a list (either of file paths or PIL Images)
        if isinstance(image_inputs, list):
            data_uris = []
            for image_input in image_inputs:
                data_uris.append(self.convert_image_to_base64(image_input))
            return data_uris
        else:
            # Single image input (file path or PIL Image)
            return self.convert_image_to_base64(image_inputs)

    def convert_image_to_base64(self, image_input):
        """Helper function to convert a single image to base64"""
        # Check if the input is a file path (string)
        if isinstance(image_input, str):
            with open(image_input, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Check if the input is a PIL Image
        elif isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")  # You can change the format if needed
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        else:
            raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

        return f"data:image/png;base64,{base64_data}"
        
    def pipeline(self, image=None, prompt: str = None):

        image = self.Tensor2PIL(image)
    
        # processor
        data_uri = self.image_to_base64_data_uri(image)
        
        # messages
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}}, 
                    {"type" : "text", "text": prompt}
                ]
            }
        ]
        
        results = self.model.create_chat_completion(messages=messages)
        
        answer = results["choices"][0]["message"]["content"].strip()
        
        return answer
    

class QwenVisionInstruct(VLMTemplate):

    def __init__(self, name=None):

        super().__init__(name=name)

        self.conversation = []
        
    def _load_weight(self):

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images=Tuple[Any], prompt: str = None):
        
        images = [self.Tensor2PIL(image) for image in images]

        self.conversation.append(
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": images[0],},
                    {"type": "image", "image": images[1],},
                    {"type": "text", "text": prompt},
                ],
            },
        )

        text = self.processor.apply_chat_template(
            self.conversation, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        ) # important to have (add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(self.conversation)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self.conversation.append(
            {
                "role": "assitant", 
                "content": output_text[0],
            },
        )
        
        return output_text[0]
        
if __name__ == "__main__":
    pass
