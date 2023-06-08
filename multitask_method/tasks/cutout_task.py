from typing import Callable, Optional

import numpy as np
from numpy import typing as npt

from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.utils import get_patch_image_slices


class Cutout(BaseTask):
    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:

        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask.shape)
        anomaly_image_shape = tuple([sample.shape[0]] + list(anomaly_mask.shape))

        sample[anomaly_patch_slices][np.broadcast_to(anomaly_mask, anomaly_image_shape)] = self.rng.random()

        return sample
