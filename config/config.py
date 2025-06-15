"""
might be merged later
"""

from config.default import cfg

def update_cfg_with_args(cfg, args):
    cfg.DATASET.ROOT_DIR = args.data_dir
    cfg.MODEL.LLM_ID = args.llm_id
    cfg.MODEL.VLM_ID = args.vlm_id
    cfg.STRATEGY.NAME = args.strategy
    cfg.DATASET.SPLIT = args.split
    cfg.EXPERIMENT.SEED = args.seed
    cfg.EXPERIMENT.RESULT_DIR = f"results/{args.strategy}_{args.split}_{args.llm_id}_{args.vlm_id}_seed{args.seed}"
    return cfg

def get_cfg_from_args(args):
    # cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file) # merge with specific yaml file
    cfg = update_cfg_with_args(cfg, args) # update from args
    cfg.freeze()
    return cfg
