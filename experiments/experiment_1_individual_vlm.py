import argparse
import logging

import sys
sys.path.append("")

import dotenv

from src.logging.logging_config import setup_logging
from src.vlm_only_reasoning.utils import load_process

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Baseline Experiment pipeline"
        )
    
    parser.add_argument(
        "--data_dir", 
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
        "--result_dir", 
        type=str,
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
            "zero-shot", "add-info-zero-shot", "VoT-zero-shot", "CoT-zero-shot",
            "CoT-prompt", "VoT-prompt",
            ],
        help="prompt type, with zero-shot, in-context, few-shot",
        required=True,
        )
    
    return parser.parse_args()

def main(args):
    setup_logging()
    logger = logging.getLogger(__name__)

    kwargs = {
        "vlm_id": args.vlm_id,
        "data_dir": args.data_dir,
        "split": args.split,
        "result_dir": args.result_dir,
        "is_shuffle": args.is_shuffle,
        "prompt_type": args.prompt_type,
    }

    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")

    pipeline = load_process(**kwargs)
    pipeline()


if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_args()  
    main(args)
