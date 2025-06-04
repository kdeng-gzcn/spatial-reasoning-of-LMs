# my_project/config/config.py
from yacs.config import CfgNode as CN


_C = CN()

### DATASET
_C.DATASET = CN()
_C.DATASET.SINGLE_DOF_CLS = CN()
_C.DATASET.OBJ_CENTERED_CLS = CN()

### MODEL
_C.MODEL = CN()
_C.MODEL.VLM = CN()
_C.MODEL.LLM = CN()

### STRATEGY
_C.STRATEGY = CN()
_C.STRATEGY.MULTI_AGENTS = CN()
_C.STRATEGY.VLM_ONLY = CN()
_C.STRATEGY.IS_TRAP = False  # whether to add trap option
_C.STRATEGY.IS_SHUFFLE = True  # whether to shuffle options

### TASK
_C.TASK = CN()
_C.TASK.NAME = "obj_centered_cls"
_C.TASK.SPLIT = "translation" # or "rotation"

### OTHER
 


# def get_cfg_defaults():
#   """Get a yacs CfgNode object with default values for my_project."""
#   # Return a clone so that the defaults will not be altered
#   # This is for the "local variable" use pattern
#   return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`