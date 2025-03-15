import argparse
import logging

import sys
sys.path.append("") # this is main.py, which needs path

from SpatialVLM.logging.logging_config import setup_logging
from SpatialVLM.Individual.utils import load_process

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Baseline Experiment pipeline"
        )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="benchmark/Rebuild_7_Scenes_1739853799", 
        help="path for data stream", 
        required=True
        )
    
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["phi", "psi", "theta", "tx", "ty", "tz", "all"], 
        default="phi", 
        help="path for data stream", 
        required=True
        )   

    parser.add_argument(
        "--vlm_id", 
        type=str, 
        default="microsoft/Phi-3.5-vision-instruct", 
        required=True
        )
    
    parser.add_argument(
        "--result_path", 
        type=str, 
        default="result/", 
        help="path for result", 
        required=True
        )
    
    parser.add_argument(
        "--is_shuffle", 
        action="store_true", 
        help="is shuffle?",
        required=False,
        )
    
    parser.add_argument(
        "--prompt_type", 
        type=str,
        choices=[
            "zero-shot", "add-info-zero-shot", "VoT-zero-shot", "CoT-zero-shot"
            ],
        default="add-info-zero-shot", 
        help="prompt type, with zero-shot, in-context, few-shot",
        required=False,
        )
    
    return parser.parse_args()

def main(args):

    # 0. make sure arg parser and logging work
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"VLM_ID: {args.vlm_id}")
    logger.info(f"DataPath: {args.data_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"ResultPath: {args.result_path}")
    logger.info(f"is_shuffle: {args.is_shuffle}")
    logger.info(f"prompt_type: {args.prompt_type}")

    # 1. Conversation Algorithm Main
    kwargs = {
        "VLM_id": args.vlm_id,
        "datapath": args.data_path,
        "subset": args.split,
        "result dir": args.result_path,
        "is_shuffle": args.is_shuffle,
        "prompt_type": args.prompt_type,
    }

    pipeline = load_process(**kwargs)
    pipeline()
    
if __name__ == "__main__":

    args = parse_args()
    
    main(args)
