import numpy as np

K = np.array(
    [[585, 0, 320],
     [0, 585, 240],
     [0, 0, 1]], 
    dtype=np.float32
)

K_inv = np.linalg.inv(K)