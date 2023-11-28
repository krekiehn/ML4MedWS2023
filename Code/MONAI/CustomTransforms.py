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