from src.models import (
    LlamaInstruct,
    Phi3VisionInstruct,
    Phi4VisionInstruct,
    QwenVisionInstruct,
    QwenInstruct,
    GPTInstruct,
    GPTVisionInstruct,
    LlavaNextVisionInstruct,
    AnthropicVisionInstruct,
    Llama4VisionInstruct,
    Idefics3VisionInstruct,
    LlavaOneVisionInstruct,
    # Gemma3VisionInstruct,
    # DeepseekVisionInstruct,
    BLIP,
    BLIPVisionInstruct,
)

def load_model(model_name):
    model_mapping = {
        # TODO: LLM
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaInstruct(name=model_name),
        "meta-llama/Llama-3.1-8B-Instruct": LlamaInstruct(name=model_name),
        "Qwen/Qwen2.5-7B-Instruct": QwenInstruct(name=model_name),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": QwenInstruct(name=model_name),
        "gpt-4": GPTInstruct(name=model_name), # llm
        "gpt-4o-text-only": GPTInstruct(name=model_name),

        # TODO: VLM
        "microsoft/Phi-3.5-vision-instruct": Phi3VisionInstruct(name=model_name), # ✅
        "microsoft/Phi-4-multimodal-instruct": Phi4VisionInstruct(name=model_name), # ❌
        "remyxai/SpaceQwen2.5-VL-3B-Instruct": QwenVisionInstruct(name=model_name), # ✅
        "Qwen/Qwen2.5-VL-7B-Instruct": QwenVisionInstruct(name=model_name), # ✅
        "Qwen/Qwen2.5-VL-32B-Instruct": QwenVisionInstruct(name=model_name), # ✅
        "Qwen/Qwen2.5-VL-72B-Instruct": QwenVisionInstruct(name=model_name), # ✅
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": Llama4VisionInstruct(name=model_name), # ✅ TODO: >>109B
        "gpt-4o-mini": GPTVisionInstruct(name=model_name), # ✅
        "gpt-4o": GPTVisionInstruct(name=model_name), # ✅
        "gpt-4-turbo": GPTVisionInstruct(name=model_name), # ✅
        "claude-sonnet-4-20250514": AnthropicVisionInstruct(name=model_name), # ❌
        # "google/gemma-3-12b-it": Gemma3VisionInstruct(name=model_name), # ❌ TODO: pkg triton
        "HuggingFaceM4/Idefics3-8B-Llama3": Idefics3VisionInstruct(name=model_name), # ✅
        # "deepseek-ai/deepseek-vl2-small": DeepseekVisionInstruct(name=model_name), # ❌ TODO: transformer==4.47.1, xformers
        "llava-hf/llama3-llava-next-8b-hf": LlavaNextVisionInstruct(name=model_name), # ✅
        "llava-hf/llava-onevision-qwen2-7b-ov-hf": LlavaOneVisionInstruct(name=model_name), # ✅
        "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": LlavaOneVisionInstruct(name=model_name), # ❌

        # TODO: BLIP
        "Salesforce/blip2-opt-2.7b": BLIP(name=model_name),
        "Salesforce/instructblip-vicuna-7b": BLIPVisionInstruct(name=model_name),
    }

    if model_name not in model_mapping:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_mapping[model_name]
