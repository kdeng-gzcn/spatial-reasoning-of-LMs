import argparse
import logging

import sys
sys.path.append("./")

from SpatialVLM.Conversation.utils import load_process

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Conversation Experiment pipeline"
        )
    
    # parser.add_argument(
    #     "--info", 
    #     type=str, 
    #     help="useless, removed", 
    #     required=False,
    #     )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./data/images_conversation", 
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
        default="llava-hf/llava-v1.6-mistral-7b-hf", 
        required=False
        )
    
    parser.add_argument(
        "--LLM", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        required=False
        )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="single", 
        choices=["single", "pair"], 
        required=False
        )
    
    parser.add_argument(
        "--result_path", 
        type=str, 
        default="./Result/Pair Conversation Experiment/", 
        help="path for result", 
        required=False
        )
    
    return parser.parse_args()

def main(args):

    # 0. make sure arg parser and logging work
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Mode: {args.mode}")
    logging.info(f"LLM: {args.LLM}")
    logging.info(f"VLM: {args.VLM}")
    logging.info(f"DataPath: {args.data_path}")
    logging.info(f"Subset: {args.subset}")
    logging.info(f"ResultPath: {args.result_path}")

    # 1. Conversation Algorithm Main
    kwargs = {
        "VLM_id": args.VLM,
        "LLM_id": args.LLM,
        "datapath": args.data_path,
        "subset": args.subset,
    }

    pipeline = load_process(type="pair", **kwargs)
    pipeline(len_conversation=1, result_dir=args.result_path)
    

if __name__ == "__main__":

    args = parse_args()
    
    main(args)
