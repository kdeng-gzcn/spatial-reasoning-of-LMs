# my_project/config/config.py
from yacs.config import CfgNode as CN


_C = CN()

### DATASET
_C.DATASET = CN()
_C.DATASET.SEVEN_SCENES = CN()
_C.DATASET.SCANNET = CN()
_C.DATASET.SCANNETPP = CN()

_C.DATASET.UTILS = CN()
_C.DATASET.UTILS.MAX_LEN_DATASET = 60 # maximum number of data points in the dataset

### MODEL
_C.MODEL = CN()
_C.MODEL.VLM = CN()
_C.MODEL.VLM.ID = "gpt-4o"  # default name for the VLM model

_C.MODEL.LLM = CN()
_C.MODEL.LLM.ID = "gpt-4o-text-only"  # default name for the LLM model

# _C.MODEL.UTILS = CN() # utility configs for model loading

### STRATEGY
_C.STRATEGY = CN()
_C.STRATEGY.MULTI_AGENTS = CN()

_C.STRATEGY.VLM_ONLY = CN()
_C.STRATEGY.VLM_ONLY.PROMPT_TYPE = "zero-shot"

_C.STRATEGY.IS_TRAP = False  # whether to add trap option
_C.STRATEGY.IS_SHUFFLE = True  # whether to shuffle options

# _C.STRATEGY.UTILS = CN()

### OTHER
_C.EXPERIMENT = CN()
_C.EXPERIMENT.TASK_NAME = ""
_C.EXPERIMENT.TASK_SPLIT = ""  # ["translation", "rotation"]
_C.EXPERIMENT.DATA_DIR = ""  # root directory of the dataset
_C.EXPERIMENT.RESULT_DIR = ""  # directory to save the results

# def get_cfg_defaults():
#   """Get a yacs CfgNode object with default values for my_project."""
#   # Return a clone so that the defaults will not be altered
#   # This is for the "local variable" use pattern
#   return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`