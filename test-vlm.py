from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

### load modules
from src.dataset.utils import load_dataset
from src.models.utils import load_model
from src.logging.logging_config import setup_logging

### load config
from config.default import cfg

### load modules
from src.prompt_generator import PromptGenerator
from src.pipeline import SpatialReasoningPipeline


def _parse_benchmark_name(benchmark_name: str) -> str:
    """
    Parse the benchmark name to get the task name and split.
    """
    parts = benchmark_name.split('-')
    if len(parts) < 2:
        raise ValueError(f"Invalid benchmark name format: {benchmark_name}")
    
    task_name = '-'.join(parts[:2]) + '-cls'
    
    return task_name


def _get_cfg():
    """
    Merge the command line arguments into the configuration.
    """
    cfg.set_new_allowed(True)  # allow new keys to be set
    cfg.EXPERIMENT.TASK_NAME = "single-dof-cls"
    cfg.STRATEGY.VLM_ONLY.PROMPT_TYPE = "zero-shot"
    cfg.EXPERIMENT.TASK_SPLIT = "theta"
    cfg.STRATEGY.IS_TRAP = False
    cfg.STRATEGY.IS_SHUFFLE = True
    return cfg


def _load_model(model_id: str):
    """
    Load the model based on the model ID.
    
    """
    model = load_model(model_id)
    model._load_weight()
    return model


def _get_benchmark_name(data_dir: str) -> str:
    """
    Extract the benchmark name from the data directory.
    """
    data_dir = Path(data_dir)
    if data_dir.is_dir():
        return data_dir.parent.name
    else:
        raise ValueError(f"Invalid data directory: {data_dir}")


def _load_dataloader(data_dir: str, cfg: Any):
    """
    Load the dataset and create a DataLoader.
    """
    dataset = load_dataset(_get_benchmark_name(data_dir), data_root_dir=data_dir, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    dataloader_tqdm = tqdm(dataloader, desc="Processing", total=len(dataloader) if hasattr(dataloader, '__len__') else None)
    return dataloader_tqdm

cfg = _get_cfg()

data_dir = "/home/u5u/kdeng.u5u/benchmark/single-dof-camera-motion-scannet/theta_significant"
dataloader = _load_dataloader(data_dir, cfg)

# Load the model
# vlm_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# vlm_id = "Qwen/Qwen2.5-VL-32B-Instruct"
# vlm_id = "Qwen/Qwen2.5-VL-72B-Instruct"
# vlm_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # TODO: >>109B
# vlm_id = "google/gemma-3-12b-it"
# vlm_id = "HuggingFaceM4/Idefics3-8B-Llama3"
vlm_id = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
# vlm_id = "deepseek-ai/deepseek-vl2-small" # TODO: transformer==4.47.1, torch==2.7.0, xformers
# vlm_id = "microsoft/Phi-3.5-vision-instruct"
# vlm_id = "microsoft/Phi-4-multimodal-instruct"
# vlm_id = "llava-hf/llama3-llava-next-8b-hf"
# vlm_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
# vlm_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
# vlm_id = "TIGER-Lab/Mantis-8B-siglip-llama3" # TODO: decord
vlm = _load_model(vlm_id)

batch = next(iter(dataloader))
item = next(iter(batch))  # get the first item in the batch

src_img, tgt_img = item["source_image"], item["target_image"]
metadata = item["metadata"]
images = (src_img, tgt_img)

vlm._clear_history()  # clear the history of VLM for each pair of images

# Load the prompt generator
prompt_generator = PromptGenerator(cfg)

# Load the pipeline
pipe = SpatialReasoningPipeline(cfg, prompt_generator=prompt_generator)

pipe.run_vlm_only(
                images=images,
                metadata=metadata,
                vlm=vlm,
                is_demo=True,
            )

# prompt = "Can you see two images? Answer the question in <ans1></ans1> tags. Are they the same images without any difference? Answer the question in <ans2></ans2> tags. Please describe the differences in details between images you see in <ans3></ans3> tags."
    
# vlm_answer = vlm.pipeline(images, prompt)

# print(f"⚠️customed VLM pipeline output:<🤖>{vlm_answer}<🤖>")