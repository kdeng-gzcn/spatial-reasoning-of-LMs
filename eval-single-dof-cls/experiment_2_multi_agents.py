import argparse
import logging

import dotenv

from src.logging.logging_config import setup_logging
from src.multi_agents_reasoning.utils import load_process

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Conversation Experiment pipeline"
        )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data/images_conversation", 
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
        required=True
        )
    
    parser.add_argument(
        "--llm_id", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        required=True
        )
    
    parser.add_argument(
        "--vlm_image_input_type",
        type=str, 
        default="pair", 
        choices=["single", "pair"], 
        required=True
        )
    
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default="./Result/Pair Conversation Experiment/", 
        help="path for result", 
        required=True
        )

    parser.add_argument(
        "--max_len_of_conv",
        type=int,
        default=5,
        help="length max_len_conv conversation",
        required=False
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

    parser.add_argument(
        "--is_remove_trap_var",
        action="store_true",
        help="choose this to remove trap variable 'no movement'",
        required=False,
    )

    parser.add_argument(
        "--is_vlm_keep_hisroty",
        action="store_true",
        help="choose this to keep history in VLM",
        required=False,
    )
    
    return parser.parse_args()


def main(args):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    kwargs = {
        "vlm_id": args.vlm_id,
        "llm_id": args.llm_id,
        "data_dir": args.data_dir,
        "split": args.split,
        "result_dir": args.result_dir,
        "is_shuffle": args.is_shuffle,
        "is_remove_trap_var": args.is_remove_trap_var,
        "prompt_type": args.prompt_type,
        "max_len_of_conv": args.max_len_of_conv,
        "vlm_image_input_type": args.vlm_image_input_type,
        "is_vlm_keep_hisroty": args.is_vlm_keep_hisroty,
    }

    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")

    pipeline = load_process(type=args.vlm_image_input_type, **kwargs)
    pipeline()


if __name__ == "__main__":
    dotenv.load_dotenv()
    args = parse_args()
    main(args)
