import os
from typing import List, Any, Tuple
import base64
import io
from PIL import Image
import logging

import torch
from torch.amp import autocast
from torchvision.transforms import ToPILImage

from openai import OpenAI

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    Idefics2Processor, Idefics2ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, # newest transformers
)

from qwen_vl_utils import process_vision_info

class VLMTemplate:
    def __init__(self, name: str):
        self.logger = logging.getLogger(__name__)

        self.Tensor2PIL = ToPILImage()
        self.model_name = name
    
    def pipeline(self, image: Tuple[Any, Any], prompt: str) -> str:
        raise NotImplementedError
    

class LlavaNextInstruct(VLMTemplate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self) -> None:
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
    def _clear_history(self) -> None:
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0: # first time
            self.conversation.extend([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Can you see this first image?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Yes, I can see the image."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ])
        else: # not first time
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        
        prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
        print(prompt)
        print()
        inputs = self.processor(images=images, text=[prompt], padding=True, return_tensors="pt").to(self.model.device)

        with autocast():      
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    )
    
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(response)
        print()
        response = response[0][len(prompt):] # need raw string to escape special characters

        self.conversation.append(
            {
                "role": "assistant", 
                "content": [
                        {"type": "text", "text": response},
                    ],
            },
        )
        return response


class Idefics2VLM(VLMTemplate):
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


class PhiVisionInstruct(VLMTemplate):
    """
    histroy func can not be used in the pipeline
    """
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            num_crops=4
        ) 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", # bfloat16
            _attn_implementation='flash_attention_2'    
        )

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
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
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device) # model.device

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

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0] 

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response


class Phi4VisionInstruct(VLMTemplate):
    """histroy func can not be used in the pipeline"""
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            num_crops=4
        ) 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", # bfloat16
            _attn_implementation='flash_attention_2'    
        )

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
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
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device) # model.device

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

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0] 

        self.conversation.append(
            {
                "role": "assitant", 
                "content": response,
            },
        )
        return response


class SpaceLLaVA(VLMTemplate):
    def __init__(self, path="/bask/projects/j/jlxi8926-auto-sum/kdeng/doppelgangers/data/doppelgangers_dataset/doppelgangers/"):
        super().__init__()
        self.model_name = "remyxai/SpaceLLaVA"
        self.model_path = path
        
    def __call__(self):
        """This maybe a __call__"""
        drive_path = self.model_path 
    
        # load pretrained weights
        mmproj = os.path.join(drive_path, "mmproj-model-f16.gguf")
        model_path = os.path.join(drive_path, "ggml-model-q4_0.gguf")
        
        # load model
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=False)
        spacellava = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers=-1, verbose=False)
        
        self.model = spacellava

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
        if isinstance(image_input, str):
            with open(image_input, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")  # You can change the format if needed
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")
        return f"data:image/png;base64,{base64_data}"
        
    def pipeline(self, image=None, prompt: str = None):
        image = self.Tensor2PIL(image)
        data_uri = self.image_to_base64_data_uri(image)
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
    def __init__(self, name):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16, # 2 bytes 
            device_map="auto",
        )

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0: # first time
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
        else: # not first time
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )
        # self.logger.error(f"Length of Conversation: {len(self.conversation)}")
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
        ).to(self.model.device) # model.device
        del image_inputs, video_inputs
        torch.cuda.empty_cache()

        with autocast(device_type="cuda"):      
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        torch.cuda.empty_cache()
        return response
    

class GPTVisionInstruct(VLMTemplate):
    def __init__(self, name):
        super().__init__(name=name)
        self.conversation = []
        self.prompt_tokens = []
        self.completion_tokens = []
        
    def _load_weight(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _clear_history(self) -> None:
        self.conversation = []

    def _calculate_input_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.15, # 0.15$ per million tokens
            "gpt-4o": 2.5, # 2.5$ per million tokens
            "gpt-4-turbo": 10, # 10$ per million tokens
        }
        assert self.model_name in cost_map, self.logger.error(f"Model {self.model_name} not found in input cost map")
        return cost_map[self.model_name] * num_tokens / 1e6
    
    def _calculate_output_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "gpt-4o-mini": 0.6, # 0.15$ per million tokens
            "gpt-4o": 10, # 2.5$ per million tokens
            "gpt-4-turbo": 30, # 30$ per million tokens
        }
        assert self.model_name in cost_map, self.logger.error(f"Model {self.model_name} not found in output cost map")
        return cost_map[self.model_name] * num_tokens / 1e6

    def _print_tokens_usage(self, completiion: Any) -> None:
        """not used"""
        self.logger.info(f"🤡 Prompt Tokens Usage: {completiion.usage.prompt_tokens}")
        self.logger.info(f"👾 Prompt Tokens Cost: {self._calculate_input_tokens_cost(completiion.usage.prompt_tokens)}")
        self.logger.info(f"🤡 Completion Tokens Usage: {completiion.usage.completion_tokens}")
        self.logger.info(f"🤡 Completion Reasoning Tokens Usage: {completiion.usage.completion_tokens_details.reasoning_tokens}")
        self.logger.info(f"👾 Completion Tokens Cost: {self._calculate_output_tokens_cost(completiion.usage.completion_tokens)}")
        self.logger.info(f"🤡 Total Tokens Usage: {completiion.usage.total_tokens}")

    def _collect_completions(self, completiion: Any) -> None:
        self.prompt_tokens.append(completiion.usage.prompt_tokens)
        self.completion_tokens.append(completiion.usage.completion_tokens)

    def print_total_tokens_usage(self) -> None:
        self.logger.info(f"😚💦💬 Total Prompt Tokens Usage: {sum(self.prompt_tokens)}")
        self.logger.info(f"💰 Total Prompt Tokens Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens))}")
        self.logger.info(f"🤖💬 Total Completion Tokens Usage: {sum(self.completion_tokens)}")
        self.logger.info(f"💰 Total Completion Tokens Cost: {self._calculate_output_tokens_cost(sum(self.completion_tokens))}")
        self.logger.info(f"🤡🤡🤡 Total Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)) + self._calculate_output_tokens_cost(sum(self.completion_tokens))}")
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        # Convert images to base64
        base64_images = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

        if len(self.conversation) == 0: # first time
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[0]}",
                            },
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[1]}",
                            },
                        },
                    ],
                },
            )
        else: # not first time
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        completiion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=1024,
            temperature=0, # temp. fixed at 0
        )
        self._collect_completions(completiion)
        # self._print_tokens_usage(completiion)
        response = completiion.choices[0].message.content

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response
