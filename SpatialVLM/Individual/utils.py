from SpatialVLM.Individual import (
    IndividualProcess,
)

def load_process(**kwargs):
    
    """
    
    kwargs:
        VLM_id
        datapath
        subset
    
    """

    return IndividualProcess(**kwargs)

