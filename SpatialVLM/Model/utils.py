from SpatialVLM.Model import (
    HuggingFaceLLM,
    Phi3VLM,
    Phi3VLMHistory,
    SpaceLLaVA,
    LlavaNextVLM,
    Idefics2VLM,
    QwenVisionInstruct,
)

def load_model(model_name):

    model_mapping = {
        "llama": HuggingFaceLLM(name="meta-llama/Meta-Llama-3-8B-Instruct"),
        "microsoft/Phi-3.5-vision-instruct": Phi3VLMHistory(name=model_name),
        "Qwen/Qwen2.5-VL-7B-Instruct": QwenVisionInstruct(name=model_name)
    }

    if model_name not in model_mapping:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_mapping[model_name]
