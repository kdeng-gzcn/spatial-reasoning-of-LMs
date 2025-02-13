from SpatialVLM.Prompter.PrompterPair import (
    TaskDesc_Prompter4Pair,
    LLM_To_VLM_Prompter4Pair,
    VLM_To_LLM_Prompter4Pair,
)

from SpatialVLM.Prompter.PrompterBaseline import (
        TaskDesc_Prompter4Baseline,
)

def load_prompter(prompter_type):

    prompter_mapping = {
        "Begin for pair": TaskDesc_Prompter4Pair,
        "Start for pair": TaskDesc_Prompter4Pair,
        "Task Description for pair": TaskDesc_Prompter4Pair,

        "Brain for pair": VLM_To_LLM_Prompter4Pair,
        "LLM for pair": VLM_To_LLM_Prompter4Pair,

        "Eye for pair": LLM_To_VLM_Prompter4Pair,
        "VLM for pair": LLM_To_VLM_Prompter4Pair,

        "Task Description for Baseline": TaskDesc_Prompter4Baseline,
    }

    if prompter_type not in prompter_mapping:
        raise NotImplementedError(f"Prompter type {prompter_type} not supported.")

    return prompter_mapping[prompter_type]()