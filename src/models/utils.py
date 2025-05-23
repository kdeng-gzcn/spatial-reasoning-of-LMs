from src.models import (
    LlamaInstruct,
    PhiVisionInstruct,
    QwenVisionInstruct,
    QwenInstruct,
    GPTInstruct,
    GPTVisionInstruct,
    LlavaNextInstruct,
)

def load_model(model_name):
    model_mapping = {
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaInstruct(name=model_name),
        "meta-llama/Llama-3.1-8B-Instruct": LlamaInstruct(name=model_name),
        "Qwen/Qwen2.5-7B-Instruct": QwenInstruct(name=model_name),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": QwenInstruct(name=model_name),
        "gpt-4": GPTInstruct(name=model_name), # llm
        # "llava-hf/llava-v1.6-mistral-7b-hf": LlavaNextInstruct(name=model_name), 

        "microsoft/Phi-3.5-vision-instruct": PhiVisionInstruct(name=model_name),
        "Qwen/Qwen2.5-VL-7B-Instruct": QwenVisionInstruct(name=model_name),
        "gpt-4o-mini": GPTVisionInstruct(name=model_name),
        "gpt-4o": GPTVisionInstruct(name=model_name),
        "gpt-4-turbo": GPTVisionInstruct(name=model_name),
    }

    if model_name not in model_mapping:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_mapping[model_name]
