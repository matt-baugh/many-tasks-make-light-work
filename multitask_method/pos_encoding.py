from abc import abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt


class PosEnc:

    def __init__(self, num_dataset_dims: int):
        self.num_dataset_dims = num_dataset_dims

    @abstractmethod
    def gen_encoding(self, sample_shape: Tuple[int, ...]) -> npt.NDArray:
        pass

    def __call__(self, sample_shape: Tuple[int, ...]) -> npt.NDArray:
        return self.gen_encoding(sample_shape)

    @abstractmethod
    def num_encoding_channels(self) -> int:
        pass


class ConvCoordEnc(PosEnc):

    def gen_encoding(self, sample_shape: Tuple[int, ...]) -> npt.NDArray:
        assert len(sample_shape) == self.num_dataset_dims

        return np.stack(np.meshgrid(*[np.linspace(-1, 1, sz) for sz in sample_shape], indexing='ij'))

    def num_encoding_channels(self) -> int:
        return self.num_dataset_dims


class GaussianRFFEnc(PosEnc):

    def __init__(self, num_dataset_dims: int, encoding_size: int, scale: float):
        super().__init__(num_dataset_dims)

        self.encoding_size = encoding_size
        self.scale = scale

        self.B = np.random.default_rng(seed=42).standard_normal((self.encoding_size, self.num_dataset_dims)) * \
            self.scale

    def gen_encoding(self, sample_shape: Tuple[int, ...]) -> npt.NDArray:
        assert len(sample_shape) == self.num_dataset_dims

        # Make initial coordinate grid, of shape (n_dims, dim_1_sz, dim_2_sz, ...)
        init_coords = np.stack(np.meshgrid(*[np.linspace(0, 1, sz, endpoint=False) for sz in sample_shape],
                                           indexing='ij'))

        # Becomes (encoding_size, dim_1_sz, dim_2_sz, ...)
        # Annoyingly matmul doesn't follow normal matrix multiplication for N>2
        proj_coords = np.einsum('ci,i...->c...', self.B, init_coords)

        scaled_coords = 2 * np.pi * proj_coords
        return np.concatenate([np.sin(scaled_coords), np.cos(scaled_coords)])

    def num_encoding_channels(self) -> int:
        # Return 2* due to sin and cos components
        return 2 * self.encoding_size


class CombinedEnc(PosEnc):

    def __init__(self, num_dataset_dims: int, rff_encoding_size: int, rff_scale: float):
        super().__init__(num_dataset_dims)
        self.cc_enc = ConvCoordEnc(num_dataset_dims)
        self.rff_enc = GaussianRFFEnc(num_dataset_dims, rff_encoding_size, rff_scale)
        self.encoders = [self.cc_enc, self.rff_enc]

    def gen_encoding(self, sample_shape: Tuple[int, ...]) -> npt.NDArray:
        return np.concatenate([enc(sample_shape) for enc in self.encoders])

    def num_encoding_channels(self) -> int:
        return sum([enc.num_encoding_channels() for enc in self.encoders])
