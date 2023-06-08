from typing import Callable, Optional

import numpy as np
from numpy import typing as npt
from scipy.ndimage import distance_transform_edt


from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.labelling import AnomalyLabeller
from multitask_method.tasks.utils import get_patch_image_slices


class SmoothIntensityChangeTask(BaseTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller], intensity_task_scale: float, **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)
        self.intensity_task_scale = intensity_task_scale

    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        num_chans = sample.shape[0]
        sample_shape = sample.shape[1:]
        num_dims = len(sample_shape)

        dist_map = distance_transform_edt(anomaly_mask)
        min_shape_dim = np.min(sample_shape)

        smooth_dist = np.minimum(min_shape_dim * (0.02 + np.random.gamma(3, 0.01)), np.max(dist_map))
        smooth_dist_map = dist_map / smooth_dist
        smooth_dist_map[smooth_dist_map > 1] = 1

        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask.shape)
        # anomaly_pixel_stds = np.array([np.std(c[anomaly_mask]) for c in sample[anomaly_patch_slices]])

        # Randomly negate, so some intensity changes are subtractions
        intensity_changes = (self.intensity_task_scale / 2 + np.random.gamma(3, self.intensity_task_scale)) \
            * np.random.choice([1, -1], size=num_chans)

        intensity_change_map = smooth_dist_map * np.reshape(intensity_changes, [-1] + [1] * num_dims)
        new_patch = sample[anomaly_patch_slices] + intensity_change_map
        spatial_axis = tuple(range(1, len(sample.shape)))
        sample[anomaly_patch_slices] = np.clip(new_patch,
                                               sample.min(axis=spatial_axis, keepdims=True),
                                               sample.max(axis=spatial_axis, keepdims=True))

        return sample
