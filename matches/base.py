import numpy as np
import torch

# info from pytorch
torch_activations = [
    "threshold", "relu", "hardtanh", "relu6", "elu", "selu", "celu", "leaky_relu",
    "prelu", "rrelu", "glu", "logsigmoid", "hardshrink", "tanhshrink", "softsign",
    "softplus", "softmin", "softmax", "softshrink", "gumbel_softmax", "log_softmax",
    "tanh", "sigmoid"
]

# can we use a GPU?
cuda_available = True if torch.cuda.device_count() > 0 else False
dtype = np.float32 if cuda_available is True else np.float64

