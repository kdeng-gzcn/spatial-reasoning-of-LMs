"""

connection between user level to development level

"""

from SpatialVLM.Conversation import (
    Conversations_Single_Image,
    Conversations_Pairwise_Image
)

def load_process(type, VLM_id=None, LLM_id=None, datapath=None):

    process_mapping = {
        "single": Conversations_Single_Image,
        "pair": Conversations_Pairwise_Image
    }

    if type not in process_mapping:
        raise NotImplementedError(f"Type of process {type} not supported.")

    return process_mapping[type]

