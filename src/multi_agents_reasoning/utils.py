from src.multi_agents_reasoning import (
    Conversations_Single_Image,
    MultiAgentsPairImageInputReasoning
)

def load_process(type, **kwargs):
    process_mapping = {
        "single": Conversations_Single_Image,
        "pair": MultiAgentsPairImageInputReasoning
    }

    if type not in process_mapping:
        raise NotImplementedError(f"Type of process {type} not supported.")

    return process_mapping[type](**kwargs)
