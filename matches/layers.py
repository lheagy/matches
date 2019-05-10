import torch
from torch import nn

from .base import torch_activations
from .utils import set_kwargs

class BaseLayer(nn.Module):

    _default_activation = "relu"

    @property
    def input_shape(self):
        return getattr(self, "_input_shape", None)

    @input_shape.setter
    def input_shape(self, val):
        """
        Shape of the layer
        """
        self._validate_shape(val)
        self._shape = val

    @property
    def output_shape(self):
        return getattr(self, "_output_shape", None)

    @output_shape.setter
    def output_shape(self, val):
        """
        Shape of the layer
        """
        self._validate_shape(val)
        self._shape = val

    def _validate_shape(self, val)
        if not isinstance(val, tuple):
            try:
                val = tuple(val)
            except:
                raise Exception(
                    f"Type of shape {type(val)} is not recognized. "
                    "Shape should be a tuple"
                )

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

    def __init__(self, input_shape=None, output_shape=None, layer=None, **kwargs):
        super(BaseLayer, self).__init__()
        set_kwargs(self, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer = layer

    def forward(self, x):
        return getattr(torch, self.activation)(self.layer(x))

class DenseLayer(BaseLayer):
    """
    A dense layer followed by an activation function
    """

    def __init__(self, input_shape=None, output_shape=None, **kwargs):
        layer = nn.Linear(input_shape, output_shape)
        super(DenseLayer, self).__init__(input_shape, output_shape)




