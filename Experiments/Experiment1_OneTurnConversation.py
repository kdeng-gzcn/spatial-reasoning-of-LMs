import argparse
import logging
# import yaml

import sys
sys.path.append("./")

# 1. import costume pipeline
from SpatialVLM.Conversation import ConversationProcess as piepline

# 2. define argument parser
def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Conversation Experiment pipeline"
        )
    
    parser.add_argument(
        "--info", 
        type=str, 
        help="test arg", 
        required=False
        )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./data/images_conversation", 
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
    
    return parser.parse_args()

# 3. define usefule function for main()
def start(args):

    logging.basicConfig(level=logging.INFO)
    
    logging.info(args.info)
    logging.info("Starting the experiment pipeline...")


# 4. main function
def main(args):

    # 0. make sure arg parser and logging work
    start(args=args)

    # 1. Conversation Algorithm Main
    if args.mode == "single": 
        pipeline = piepline.Conversations_Single_Image(
            VLM_id=args.VLM, 
            LLM_id=args.LLM, 
            datapath=args.data_path
            )
        pipeline(len_conversation=1)
    
    if args.mode == "pair":
        pipeline = piepline.Conversations_Pairwise_Image(
            VLM_id=args.VLM, 
            LLM_id=args.LLM, 
            datapath=args.data_path
            )
        pipeline(len_conversation=1)
    

if __name__ == "__main__":
    
    # 0. get args from bash
    args = parse_args()
    
    # 1. use args to run custom pipeline
    main(args)
