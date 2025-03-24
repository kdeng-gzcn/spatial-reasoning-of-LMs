from src.ans_parser import (
    Metric012,
    Metric012Baseline,
    Metric0123Baseline,
    Metric0123Conv,
)

def load_ans_parser(metric_type, **kwargs):
    metric_mapping = {
        "012": Metric012,
        "Reasoning Parser for Pair Image Input": Metric0123Conv,
        
        "Baseline Metric 012": Metric012Baseline,
        "Answer Parser for VLM-Only Reasoning": Metric0123Baseline,
    }

    if metric_type not in metric_mapping:
        raise NotImplementedError(f"Metric type {metric_type} not supported.")

    return metric_mapping[metric_type](**kwargs)