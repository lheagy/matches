import torch
from torch import nn


class DenseNet(nn.Module):

    latent_dim = 2  #: output size of the bottleneck layer
    layer_dims = [64, 32, 16]  #: list of layer dimensions"
    layer_activation = "relu"  #: activation function for the dense layers
    bottleneck_activation = "tanh"  #: activation function for the bottleneck layer
    output_activation = "tanh"  #: activation function for the final output layer
    batchnorm_ind = 1  #: location in the encoding or decoding for a batchnormalization step

    def __init__(self, data_loader=None, **kwargs):
        super().__init__()

        self.input_shape = (data_loader.batch_size,)+tuple(data_loader.dataset.tensors[0].shape[1:])
        self.output_shape = (training_dl.batch_size,)+tuple(data_loader.dataset.tensors[1].shape[1:])
        set_kwargs(self, **kwargs)

        # encoding / decoding
        input_shapes = [np.prod(self.input_shape)] + self.layer_dims
        output_shapes = [self.latent_dim] + self.layer_dims[::-1]
        encoding = []
        decoding = []

        for i in range(len(self.layer_dims)):
            encoding += [DenseLayer(shape=input_shapes[i:i+2], activation=self.layer_activation)]
            decoding += [DenseLayer(shape=output_shapes[i:i+2], activation=self.layer_activation)]

            if self.batchnorm_ind is not None:
                if self.batchnorm_ind == i+1:
                    encoding += [nn.BatchNorm1d(input_shapes[i+1])]
                    decoding += [nn.BatchNorm1d(output_shapes[i+1])]

        self.encoding = nn.ModuleList(encoding)

        # bottleneck
        self.bottleneck = DenseLayer(
            shape=(self.layer_dims[-1], self.latent_dim),
            activation=self.bottleneck_activation
        )

        self.decoding = nn.ModuleList(decoding)

        # output_layer
        self.output = DenseLayer(
            shape=(output_shapes[-1], np.prod(self.output_shape)),
            activation=self.output_activation
        )


    def encode(self, x):
        y = x.flatten()
        for enc in self.encoding:
            y = enc(y)

        return self.bottleneck(y)

    def decode(self, x):
        for dec in self.decoding:
            y = dec(y)

        y = self.output(y)

        return(self.output_shape)

    def forward(self, x):
        return self.decode(self.encode(x))
