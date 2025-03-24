import numpy as np

class PromptTemplate:
    def __init__(self) -> None:
        seed = 42
        np.random.seed(seed)

    def __call__(self):
        raise NotImplementedError()
