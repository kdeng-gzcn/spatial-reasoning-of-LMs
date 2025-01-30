from SpatialVLM.Prompter import StartPrompt, BrainPrompt

def load_prompter(prompter_type):

    prompter_mapping = {
        "Begin": StartPrompt,
        "Start": StartPrompt,
        "Task Description": StartPrompt,
        "Brain": BrainPrompt,
        "LLM": BrainPrompt,
    }

    if prompter_type not in prompter_mapping:
        raise NotImplementedError(f"Prompter type {prompter_type} not supported.")

    return prompter_mapping[prompter_type]