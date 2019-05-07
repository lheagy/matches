import torch
from torch import nn

from .base import torch_activations
from .utils import set_kwargs

class DenseLayer(nn.Module):
    """
    A dense layer followed by an activation function
    """

    _default_activation = "relu"

    @property
    def shape(self):
        return getattr(self, "_shape", None)

    @shape.setter
    def shape(self, val):
        """
        Shape of the layer
        """
        if not isinstance(val, tuple):
            try:
                val = tuple(val)
            except:
                raise Exception(
                    f"Type of shape {type(shape)} is not recognized. "
                    "Shape should be a tuple"
                )
        self._shape = val

    @property
    def activation(self):
        """
        activation function on the layer
        """
        return getattr(self, "_activation", self._default_activation)

    @activation.setter
    def activation(self, val):
        if val not in torch_activations:
            raise Exception(
                f"Unknown activation: {val}. "
                "Activation function must be one of the pytorch activations."
           )
        self._activation = val

    def __init__(self, **kwargs):
        super(DenseLayer, self).__init__()
        set_kwargs(self, **kwargs)
        self.layer = nn.Linear(self.shape[0], self.shape[1])

    def forward(self, x):
        return getattr(torch, self.activation)(self.layer(x))
