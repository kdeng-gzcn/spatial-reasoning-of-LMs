from SpatialVLM.Metric import (
    Metric012,
    Metric012Baseline,
)

def load_metric(metric_type, **kwargs):

    metric_mapping = {
        "012": Metric012,
        "Baseline Metric": Metric012Baseline,
    }

    if metric_type not in metric_mapping:
        raise NotImplementedError(f"Metric type {metric_type} not supported.")

    return metric_mapping[metric_type](**kwargs)