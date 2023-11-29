import numpy as np
import monai.transforms as T


class ReplaceValuesNotInList(T.MapTransform):
    def __init__(self, keys, allowed_values, replacement_value):
        super().__init__(keys)
        self.allowed_values = set(allowed_values)
        self.replacement_value = replacement_value

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = np.where(np.isin(data[key], list(self.allowed_values)),
                                     data[key],
                                     self.replacement_value)
        return data


class PadToMaxSize(T.MapTransform):
    def __init__(self, keys, mode='constant', constant_values=0):
        super().__init__(keys)
        # self.keys = keys
        self.mode = mode
        self.constant_values = constant_values

    def __call__(self, data):
        # Calculate the maximum spatial size for each dimension
        for key in self.keys:
            if key in data:
                print(key, data[key].shape)

        max_sizes = np.max([np.array(data[key].shape) for key in self.keys], axis=0)
        print('max size', max_sizes)
        # Calculate the padding amounts for each dimension
        padding = {}
        for key in self.keys:
            padding[key] = [(0, max_size - data[key].shape[i]) for i, max_size in enumerate(max_sizes)]
        print('padding', padding)

        # Pad each image in the batch
        for key in self.keys:
            if key in data:
                data[key] = np.pad(data[key], padding[key], mode=self.mode, constant_values=self.constant_values)

        return data