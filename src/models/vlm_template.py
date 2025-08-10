import os
from typing import List, Any, Tuple, Dict
import base64
import io
import logging

import torch
from torchvision.transforms import ToPILImage

from openai import OpenAI
import anthropic

from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    GenerationConfig, # for Phi4VisionInstruct
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    # Gemma3ForConditionalGeneration,
    AutoModelForVision2Seq,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images


class VLMTemplate:
    """
    easy template
    """
    def __init__(self, name: str):
        self.logger = logging.getLogger(__name__)

        self.Tensor2PIL = ToPILImage()
        self.model_name = name
    
    def pipeline(self, image: Tuple[Any, Any], prompt: str) -> str:
        raise NotImplementedError

## Old VLM
class BLIP(VLMTemplate):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.conversation = []

    def _load_weight(self):
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
        )

    def _clear_history(self):
        self.conversation = []
    
    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
        )

        output_text = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=True,
        )
        response = output_text.strip()

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response
        

class BLIPVisionInstruct(VLMTemplate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []

    def _load_weight(self) -> None:
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name, 
            device_map="auto",
        )

    def _clear_history(self) -> None:
        self.conversation = []
    
    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
        )
        response = output_text[0].strip()
        response = response[len(prompt):].strip()

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response

## Open source models (multi image via conversation input)
# TODO

## Open source models (multi image with text input)
class LlavaNextVisionInstruct(VLMTemplate):
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
        
        text = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=images, 
            text=text,
            # padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        
        response = self.processor.decode(
            output_ids[:, inputs["input_ids"].shape[-1]:][0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True,
        )
        response = response.strip()

        self.conversation.append(
            {
                "role": "assistant", 
                "content": [
                        {"type": "text", "text": response},
                    ],
            },
        )
        return response


class LlavaOneVisionInstruct(VLMTemplate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []
        
    def _load_weight(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto",
        )

    def _clear_history(self) -> None:
        self.conversation = []

    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        self.conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        text = self.processor.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        self.conversation.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        return response

    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0: # first time
            self.conversation.extend([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]},
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
        
        inputs = self.processor.apply_chat_template(
            self.conversation, 
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            # do_sample=False,
            )
        
        response = self.processor.decode(
            output_ids[:, inputs["input_ids"].shape[-1]:][0], 
            skip_special_tokens=True, 
        )
        response = response.strip()

        self.conversation.append(
            {
                "role": "assistant", 
                "content": [
                        {"type": "text", "text": response},
                    ],
            },
        )
        return response


class Idefics3VisionInstruct(VLMTemplate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []

    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[torch.Tensor, torch.Tensor], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        text = self.processor.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text, 
            images=images, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response

    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        self.conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        text = self.processor.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        self.conversation.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        return response


# class DeepseekVisionInstruct(VLMTemplate):
#     def __init__(self, name: str):
#         super().__init__(name=name)
#         self.conversation = []

#     def _load_weight(self):
#         self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
#         self.tokenizer = self.processor.tokenizer
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name, 
#             trust_remote_code=True,
#         )
#         self.model = self.model.to(torch.bfloat16).cuda().eval()

#     def _clear_history(self):
#         self.conversation = []
        
#     def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
#         images = [self.Tensor2PIL(image) for image in images]
#         if len(self.conversation) == 0:
#             self.conversation.append(
#                 {
#                     "role": "<|User|>",
#                     "content": "<image>The first image you see is from source viewpoint, "
#                     "<image>The second image you see is from target viewpoint, " 
#                     + prompt,
#                     "images": [
#                         "images/source_image.png",
#                         "images/target_image.png",
#                     ],
#                 }
#             )
#             self.conversation.append(
#                 {
#                     "role": "<|Assistant|>", 
#                     "content": ""
#                 }
#             )
#         else:
#             self.conversation.append(
#                 {
#                     "role": "user", 
#                     "content": prompt,
#                 }
#             )
#             self.conversation.append(
#                 {"role": "Assistant", "content": ""}
#             )

#         prepare_inputs = self.processor(
#             conversations=self.conversation,
#             images=images,
#             force_batchify=True,
#             system_prompt=""
#         ).to(self.model.device)

#         inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

#         outputs = self.model.language_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=prepare_inputs.attention_mask,
#             pad_token_id=self.tokenizer.eos_token_id,
#             bos_token_id=self.tokenizer.bos_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             max_new_tokens=1024,
#             do_sample=False,
#             use_cache=True
#         )

#         answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
#         print(f"‼️demo\n{prepare_inputs['sft_format'][0]}", answer, "\ndemo‼️")

#         response = answer
#         self.conversation.append(
#             {
#                 "role": "assistant", 
#                 "content": response,
#             },
#         )
#         return response


class Phi3VisionInstruct(VLMTemplate):
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
            num_crops=16,
            use_fast=True,
        ) 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype="auto", # bfloat16
            _attn_implementation='flash_attention_2'    
        )

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0:
            prompt = "<|image_1|>\n<|image_2|>\n" + prompt
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt
                },
            )
        else:
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
            "do_sample": False,
            # "temperature": 0.0, 
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
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2',
        ).cuda()
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(
            self.model_name
        )

        # Define prompt structure
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

    def _clear_history(self):
        self.conversation = []
        
    def pipeline(self, images: Tuple[Any, Any], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        if len(self.conversation) == 0:
            text = f'{user_prompt}<|image_1|>\n<|image_2|>\n{prompt}{prompt_suffix}{assistant_prompt}'
            self.conversation.append(
                {
                    "role": "user", 
                    "content": text
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": text,
                },
            )
        inputs = self.processor(text=text, images=images, return_tensors='pt').to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            generation_config=self.generation_config,
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        self.conversation.append(
            {
                "role": "assitant", 
                "content": response,
            },
        )
        return response


class Llama4VisionInstruct(VLMTemplate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.conversation = []

    def _load_weight(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16, # 2 bytes
        )

    def _clear_history(self):
        self.conversation = []

    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        text = self.processor.apply_chat_template(
            self.conversation, 
            add_generation_prompt=True, 
            tokenize=False,
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image], 
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=1024,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response

    def pipeline(self, images: Tuple[torch.Tensor, torch.Tensor], prompt: str) -> str:
        images = [self.Tensor2PIL(image) for image in images]
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        inputs = self.processor.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response


# class Gemma3VisionInstruct(VLMTemplate):
#     def __init__(self, name: str):
#         super().__init__(name=name)
#         self.conversation = [
#             {
#                 "role": "system",
#                 "content": [
#                     {"type": "text", "text": "You are a helpful assistant."}
#                 ]
#             },
#         ]
        
#     def _load_weight(self):
#         self.processor = AutoProcessor.from_pretrained(
#             self.model_name,
#             padding_side="left",
#         )
#         self.model = Gemma3ForConditionalGeneration.from_pretrained(
#             self.model_name, 
#             device_map="auto",
#             torch_dtype=torch.bfloat16, # 2 bytes
#             attn_implementation="sdpa",
#         ).eval()

#     def _clear_history(self):
#         self.conversation = []
        
#     def pipeline(self, images: Tuple[torch.Tensor, torch.Tensor], prompt: str) -> str:
#         images = [self.Tensor2PIL(image) for image in images]
#         if len(self.conversation) == 0:
#             self.conversation.append(
#                 {
#                     "role": "user", 
#                     "content": [
#                         {"type": "image", "image": images[0]},
#                         {"type": "image", "image": images[1]},
#                         {"type": "text", "text": prompt},
#                     ],
#                 },
#             )
#         else:
#             self.conversation.append(
#                 {
#                     "role": "user", 
#                     "content": prompt,
#                 },
#             )

#         inputs = self.processor.apply_chat_template(
#             self.conversation, 
#             add_generation_prompt=True, 
#             tokenize=True,
#             return_dict=True, 
#             return_tensors="pt", 
#             do_pan_and_scan=True, # important for input images
#         ).to(self.model.device, dtype=torch.bfloat16)

#         input_len = inputs["input_ids"].shape[-1]

#         generation = self.model.generate(
#             **inputs, 
#             max_new_tokens=1024, 
#             do_sample=False
#         )
#         generation = generation[0][input_len:]

#         decoded = self.processor.decode(generation, skip_special_tokens=True)
#         response = decoded

#         self.conversation.append(
#             {
#                 "role": "assistant", 
#                 "content": response,
#             },
#         )
#         return response
    

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
            # torch_dtype="auto", 
            device_map="auto",
        )

    def _clear_history(self):
        self.conversation = []
    
    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        if len(self.conversation) == 0:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            self.conversation.append(
                {
                    "role": "user", 
                    "content": prompt,
                },
            )

        text = self.processor.apply_chat_template(
            self.conversation, 
            add_generation_prompt=True, 
            tokenize=False,
        )
        
        inputs = self.processor(
            text=[text], 
            images=[image], 
            return_tensors="pt",
        ).to(self.model.device)

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
        return response
        
    def pipeline(self, images: Tuple[torch.Tensor, torch.Tensor], prompt: str) -> str:
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
        ).to(self.model.device)

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
    
    
## Proprietary models
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

    def pipe_one_img(self, image: torch.Tensor, prompt: str) -> str:
        image = self.Tensor2PIL(image)
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
                                "url": f"data:image/jpeg;base64,{base64_image}",
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
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=1024,
            temperature=0, # temp. fixed at 0
        )
        self._collect_completions(completion)
        # self._print_tokens_usage(completiion)
        response = completion.choices[0].message.content
        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response
        
    def pipeline(self, images: Tuple[torch.Tensor, torch.Tensor], prompt: str) -> str:
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

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=1024,
            temperature=0, # temp. fixed at 0
        )
        self._collect_completions(completion)
        # self._print_tokens_usage(completiion)
        response = completion.choices[0].message.content

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response


class AnthropicVisionInstruct(VLMTemplate):
    def __init__(self, name):
        super().__init__(name=name)
        self.conversation = []
        self.prompt_tokens = []
        self.output_tokens = []
        
    def _load_weight(self) -> None:
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _clear_history(self) -> None:
        self.conversation = []

    def _calculate_input_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "claude-sonnet-4-20250514": 3, # 3$ per million tokens
            "claude-opus-4-20250514": 15, # 15$ per million tokens
        }
        assert self.model_name in cost_map, self.logger.error(f"Model {self.model_name} not found in input cost map")
        return cost_map[self.model_name] * num_tokens / 1e6
    
    def _calculate_output_tokens_cost(self, num_tokens: int) -> float:
        cost_map = {
            "claude-sonnet-4-20250514": 15, # 15$ per million tokens
            "claude-opus-4-20250514": 75, # 75$ per million tokens
        }
        assert self.model_name in cost_map, self.logger.error(f"Model {self.model_name} not found in output cost map")
        return cost_map[self.model_name] * num_tokens / 1e6

    def _collect_tokens_count(self, response: str) -> None:
        input_tokens = self.client.messages.count_tokens(
            model=self.model_name,
            messages=self.conversation
        )
        self.prompt_tokens.append(input_tokens.input_tokens)
        output_tokens = self.client.messages.count_tokens(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": response
            }],
        )
        self.output_tokens.append(output_tokens.input_tokens)

    def print_total_tokens_usage(self) -> None:
        self.logger.info(f"😚💦💬 Total Prompt Tokens Usage: {sum(self.prompt_tokens)}")
        self.logger.info(f"💰 Total Prompt Tokens Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)):.3f}$")
        self.logger.info(f"🤖💬 Total Completion Tokens Usage: {sum(self.output_tokens)}")
        self.logger.info(f"💰 Total Completion Tokens Cost: {self._calculate_output_tokens_cost(sum(self.output_tokens)):.3f}$")
        self.logger.critical(f"🤡🤡🤡 Total Cost: {self._calculate_input_tokens_cost(sum(self.prompt_tokens)) + self._calculate_output_tokens_cost(sum(self.output_tokens)):.3f}$")
        
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
                            "text": "Image 1:"
                        },
                        {
                            "type": "image", 
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_images[0],
                            },
                        },
                        {
                            "type": "text",
                            "text": "Image 2:"
                        },
                        {
                            "type": "image", 
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_images[1],
                            },
                        },
                        {
                            "type": "text", 
                            "text": prompt
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

        message = self.client.messages.create(
            model=self.model_name,
            messages=self.conversation,
            max_tokens=1024,
            temperature=0,
        )
        response = message.content[0].text
        self._collect_tokens_count(response)

        self.conversation.append(
            {
                "role": "assistant", 
                "content": response,
            },
        )
        return response