from src.models import (
    LlamaInstruct,
    PhiVisionInstruct,
    QwenVisionInstruct,
    QwenInstruct,
    Phi4VisionInstruct,
)

def load_model(model_name):
    model_mapping = {
        "meta-llama/Meta-Llama-3-8B-Instruct": LlamaInstruct(name=model_name),
        "microsoft/Phi-3.5-vision-instruct": PhiVisionInstruct(name=model_name),
        "Qwen/Qwen2.5-VL-7B-Instruct": QwenVisionInstruct(name=model_name),
        "Qwen/Qwen2.5-7B-Instruct": QwenInstruct(name=model_name),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": QwenInstruct(name=model_name),
        # "microsoft/Phi-4-multimodal-instruct": PhiVisionInstruct(name=model_name),
    }

    if model_name not in model_mapping:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_mapping[model_name]
