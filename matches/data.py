import numpy as np
from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

from .base import dtype


def data_dict_to_torch_data_loader(
    data_dict, input_keys, output_keys, batch_size=1024, drop_last=True
):
    """
    Create a torch data loader from a dictionary with numpy arrays as values
    """

    def numpy_tensor(keys):
        data_shape = data_dict[keys[0]].shape
        dim = len(data_shape)

        if not all([data_dict[k].shape == data_shape for k in keys]):
            raise Exception("Dimension mismatch for {}".format(keys))

        return np.squeeze(  # remove unnecessary dimensionality (e.g. if a dimension length 1)
            np.stack(
                [data_dict[k] for k in keys], axis=dim
            )
        ).astype(dtype)


    inputs = numpy_tensor(input_keys)
    outputs = numpy_tensor(output_keys)

    dataset = TensorDataset(from_numpy(inputs), from_numpy(outputs))

    loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last
    )

    return loader
