import torch
from random import choices
from typing import Tuple, List, Optional
from torch import Tensor
import numbers
import warnings


class RandomErasingVector(torch.nn.Module):
    """
    Inspired by: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
    Randomly selects a 'rectangle' region in a weight vector and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input data.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased checkpoint.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), value=0, mode="block"):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if scale[0] > scale[1]:
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        if not (mode == "block" or mode == "scatter"):
            raise ValueError("vector erase mode should be <<block>> or <<scatter>>")

        self.p = p
        self.scale = scale
        self.value = value
        self.mode = mode

    @staticmethod
    def erase_vector_block(
        vector: Tensor,
        scale: Tuple[float, float],
        value: Optional[List[float]] = None,
    ) -> Tensor:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (tuple or list): range of proportion of erased area against input image.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            erased vector.
        """
        len = vector.shape[0]

        # attempt 10 times
        for _ in range(10):
            erase_len = int(len * torch.empty(1).uniform_(scale[0], scale[1]).item())
            index_begin = torch.randint(0, len, size=(1,)).item()

            if not (index_begin + erase_len < len):
                continue

            if value is None:
                v = torch.empty(vector.shape, dtype=torch.float32).normal_()
            else:
                v = torch.ones(vector.shape) * value

            vector[index_begin : index_begin + erase_len] = v[
                index_begin : index_begin + erase_len
            ]
            return vector

        # Return original image
        return vector

    @staticmethod
    def erase_vector_scatter(
        vector: Tensor, scale: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (tuple or list): range of proportion of erased area against input image.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            erased vector.
        """
        len = vector.shape[0]
        erase_len = int(len * torch.empty(1).uniform_(scale[0], scale[1]).item())
        index_erase = choices(list(range(len)), k=erase_len)

        if value is None:
            v = torch.empty(vector.shape, dtype=torch.float32).normal_()
        else:
            v = torch.ones(vector.shape) * value

        vector[index_erase] = v[index_erase]

        return vector

    def forward(self, vector):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = self.value
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if self.mode == "block":
                vector = self.erase_vector_block(vector, self.scale, value)
            elif self.mode == "scatter":
                vector = self.erase_vector_scatter(vector, self.scale, value)
            return vector
        return vector
