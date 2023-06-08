import functools
import itertools
from typing import Callable, Tuple, Optional

import numpy as np
from numpy import typing as npt
from scipy.ndimage import affine_transform

from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.blending_methods import cut_paste, patch_interpolation, poisson_image_editing
from multitask_method.tasks.labelling import AnomalyLabeller
from multitask_method.tasks.utils import accumulate_rotation, accumulate_scaling,  get_patch_image_slices


class BasePatchBlendingTask(BaseTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]],
                 blend_images: Callable[[npt.NDArray[float], npt.NDArray[float], npt.NDArray[int], npt.NDArray[bool]],
                                        npt.NDArray[float]],
                 **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)
        self.get_source_sample_and_mask = get_source_sample_and_mask
        self.blend_images = blend_images

    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        num_channels = sample.shape[0]
        num_dims = len(sample.shape[1:])

        # Sample source to blend into current sample
        source_sample, source_sample_mask = self.get_source_sample_and_mask()
        source_sample_shape = np.array(source_sample.shape[1:])
        assert len(source_sample_shape) == num_dims, 'Source and target have different number of spatial dimensions: ' \
                                                     f's-{len(source_sample_shape)}, t-{num_dims}'
        assert source_sample.shape[0] == num_channels, \
            f'Source and target have different number of channels: s-{source_sample.shape[0]}, t-{num_channels}'

        # Compute INVERSE transformation matrix for parameters (rotation, resizing)
        # This is the backwards operation (final source region -> initial source region).

        trans_matrix = functools.reduce(lambda m, ds: accumulate_rotation(m,
                                                                          self.rng.uniform(-np.pi / 4, np.pi / 4),
                                                                          ds),
                                        itertools.combinations(range(num_dims), 2),
                                        np.identity(num_dims))

        # Compute effect on corner coords
        target_anomaly_shape = np.array(anomaly_mask.shape)
        corner_coords = np.array(np.meshgrid(*np.stack([np.zeros(num_dims), target_anomaly_shape], axis=-1),
                                             indexing='ij')).reshape(num_dims, 2 ** num_dims)

        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        init_grid_shape = max_trans_coords - min_trans_coords

        # Sample scale and clip so that source region isn't too big
        max_scale = np.min(0.8 * source_sample_shape / init_grid_shape)

        # Compute final transformation matrix
        scale_change = 1 + self.rng.exponential(scale=0.1)
        scale_raw = self.rng.choice([scale_change, 1 / scale_change])
        scale = np.minimum(scale_raw, max_scale)

        trans_matrix = accumulate_scaling(trans_matrix, scale)

        # Recompute effect on corner coord
        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        final_init_grid_shape = max_trans_coords - min_trans_coords

        if np.any(final_init_grid_shape > source_sample_shape):
            print('Source shape: ', source_sample_shape)
            print('Source extracted shape without scale:, ', init_grid_shape)
            print('Resize factor: ', scale)
            print('Final extracted shape:, ', final_init_grid_shape)
            print()

        # Choose anomaly source location
        final_init_grid_shape = final_init_grid_shape.astype(int)
        min_corner = np.zeros(len(source_sample_shape))
        max_corner = source_sample_shape - final_init_grid_shape

        while True:
            source_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

            # Don't want to actually calculate mask overlap, as would need to apply all transformations to mask in
            # reverse, which would take time. Better use heuristic that, if middle of patch is in mask, then probably
            # a good amount (>50%) of patch is covered by object
            if source_sample_mask is None or source_sample_mask[tuple(source_corner + final_init_grid_shape // 2)]:
                break

        # Extract source
        source_orig = source_sample[get_patch_image_slices(source_corner, tuple(final_init_grid_shape))]

        # Because we computed the backwards transformation we don't need to inverse the matrix
        source_to_blend = np.stack([affine_transform(chan, trans_matrix, offset=-min_trans_coords,
                                                     output_shape=tuple(target_anomaly_shape))
                                    for chan in source_orig])

        spatial_axis = tuple(range(1, len(source_sample.shape)))
        # Spline interpolation can make values fall outside domain, so clip to the original range
        source_to_blend = np.clip(source_to_blend,
                                  source_sample.min(axis=spatial_axis, keepdims=True),
                                  source_sample.max(axis=spatial_axis, keepdims=True))

        # As the blending can alter areas outside the mask, update the mask with any effected areas
        aug_sample = self.blend_images(sample, source_to_blend, anomaly_corner, anomaly_mask)
        sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))
        sample_diff = np.mean(np.abs(sample[sample_slices] - aug_sample[sample_slices]), axis=0)
        anomaly_mask[sample_diff > 0.001] = True

        # Return sample with source blended into it
        return aug_sample


class TestCutPastePatchBlender(BasePatchBlendingTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: npt.NDArray[float],
                 test_source_sample_mask: Optional[npt.NDArray[bool]], **kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask), cut_paste)


class TestPatchInterpolationBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: npt.NDArray[float],
                 test_source_sample_mask: Optional[npt.NDArray[bool]], **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask), patch_interpolation,
                         **all_kwargs)


class TestPoissonImageEditingMixedGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: npt.NDArray[float],
                 test_source_sample_mask: Optional[npt.NDArray[bool]],
                 **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask),
                         lambda *args: poisson_image_editing(*args, True),
                         **all_kwargs)


class TestPoissonImageEditingSourceGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: npt.NDArray[float],
                 test_source_sample_mask: Optional[npt.NDArray[bool]], **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask),
                         lambda *args: poisson_image_editing(*args, False),
                         **all_kwargs)


class PoissonImageEditingMixedGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]],
                 **all_kwargs):
        super().__init__(sample_labeller,
                         get_source_sample_and_mask,
                         lambda *args: poisson_image_editing(*args, True),
                         **all_kwargs)


class PoissonImageEditingSourceGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]],
                 **all_kwargs):
        super().__init__(sample_labeller,
                         get_source_sample_and_mask,
                         lambda *args: poisson_image_editing(*args, False),
                         **all_kwargs)
