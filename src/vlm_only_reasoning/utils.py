from src.vlm_only_reasoning import (
    VLMOnlyReasoning,
)

def load_process(**kwargs):
    """
    kwargs:
        VLM_id
        datapath
        split
        result_dir
    """

    return VLMOnlyReasoning(**kwargs)
