from SpatialVLM.Individual import (
    IndividualProcess,
)

def load_process(**kwargs):
    """
    kwargs:
        VLM_id
        datapath
        split
        result_dir
    """

    return IndividualProcess(**kwargs)
