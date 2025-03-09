import argparse
import logging

import sys
sys.path.append("./")

from SpatialVLM.logging.logging_config import setup_logging
from SpatialVLM.Individual.utils import load_process

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Baseline Experiment pipeline"
        )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./data/Rebuild_7_Scenes_1200_1738445186", 
        help="path for data stream", 
        required=False
        )
    
    parser.add_argument(
        "--subset", 
        type=str, 
        default="phi", 
        help="path for data stream", 
        required=False
        )

    parser.add_argument(
        "--VLM", 
        type=str, 
        default="???", 
        required=False
        )
    
    parser.add_argument(
        "--result_path", 
        type=str, 
        default="./Result/Individual VLM Experiment/", 
        help="path for result", 
        required=False
        )
    
    return parser.parse_args()

def main(args):

    # 0. make sure arg parser and logging work
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"VLM: {args.VLM}")
    logger.info(f"DataPath: {args.data_path}")
    logger.info(f"Subset: {args.subset}")
    logger.info(f"ResultPath: {args.result_path}")

    # 1. Conversation Algorithm Main
    kwargs = {
        "VLM_id": args.VLM,
        "datapath": args.data_path,
        "subset": args.subset,
        "result dir": args.result_path,
    }

    pipeline = load_process(**kwargs)
    pipeline()
    
if __name__ == "__main__":

    args = parse_args()
    
    main(args)
