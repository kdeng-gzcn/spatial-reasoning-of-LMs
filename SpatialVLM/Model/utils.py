from SpatialVLM.Model import (
    HuggingFaceLLM,
    Phi3VLM,
    SpaceLLaVA,
    LlavaNextVLM,
    Idefics2VLM,
)

def load_model(model_name):

    model_mapping = {
        "llama": HuggingFaceLLM(name="meta-llama/Meta-Llama-3-8B-Instruct"),
        "Phi 3.5": Phi3VLM(name="microsoft/Phi-3.5-vision-instruct"),
    }

    if model_name not in model_mapping:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_mapping[model_name]
