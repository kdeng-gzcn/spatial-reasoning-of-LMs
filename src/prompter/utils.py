from src.prompter.PrompterPair import (
    TaskPrompterPairImageInput,
    LLMQuestionToVLMPairImageInput,
    VLMAnswerToLLMPairImageInput,
)

from src.prompter.PrompterBaseline import (
        TaskPrompterVLMOnly,
)

def load_prompter(prompter_type, **kwargs):
    prompter_mapping = {
        "Task Prompt for Pair Image Input": TaskPrompterPairImageInput,
        "Spatial Understanding Question Prompt for Pair Image Input": LLMQuestionToVLMPairImageInput,
        "Spatial Reasoning Prompt for Pair Image Input": VLMAnswerToLLMPairImageInput,

        "Task Description for Baseline": TaskPrompterVLMOnly,
    }

    if prompter_type not in prompter_mapping:
        raise NotImplementedError(f"Prompter type {prompter_type} not supported.")

    return prompter_mapping[prompter_type](**kwargs)
