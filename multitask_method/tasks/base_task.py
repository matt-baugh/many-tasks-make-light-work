from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from multitask_method.tasks.labelling import AnomalyLabeller
from multitask_method.tasks.task_shape import EitherDeformedHypershapePatchMaker
from multitask_method.tasks.utils import get_patch_slices, nsa_sample_dimension


class BaseTask(ABC):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller] = None, **all_kwargs):
        self.sample_labeller = sample_labeller
        self.rng = np.random.default_rng()
        self.anomaly_shape_maker = EitherDeformedHypershapePatchMaker(nsa_sample_dimension)
        self.all_kwargs = all_kwargs

    def apply(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]], *args, **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :return: sample with task applied and label map.
        """

        aug_sample = sample.copy()
        sample_shape = np.array(sample.shape[1:])
        anomaly_mask = np.zeros(sample_shape, dtype=bool)

        min_anom_prop = self.all_kwargs.get('min_anom_prop', 0.06)
        max_anom_prop = self.all_kwargs.get('max_anom_prop', 0.8)
        min_dim_lens = (min_anom_prop * sample_shape).round().astype(int)
        max_dim_lens = (max_anom_prop * sample_shape).round().astype(int)
        dim_bounds = list(zip(min_dim_lens, max_dim_lens))

        # For random number of times
        for i in range(4):

            # Compute anomaly mask
            curr_anomaly_mask, intersect_fn = self.anomaly_shape_maker.get_patch_mask_and_intersect_fn(dim_bounds,
                                                                                                       sample_shape)
            # Choose anomaly location
            anomaly_corner = self.find_valid_anomaly_location(curr_anomaly_mask, sample_mask, sample_shape)

            # Apply self-supervised task
            aug_sample = self.augment_sample(aug_sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn)
            anomaly_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)] |= curr_anomaly_mask

            # Randomly brake at end of loop, ensuring we get at least 1 anomaly
            if self.rng.random() > 0.5:
                break

        if self.sample_labeller is not None:
            return aug_sample, self.sample_labeller(aug_sample, sample, anomaly_mask)
        else:
            # If no labeller is provided, we are probably in a calibration process
            return aug_sample, np.expand_dims(anomaly_mask, 0)

    def find_valid_anomaly_location(self, curr_anomaly_mask: npt.NDArray[bool],
                                    sample_mask: Optional[npt.NDArray[bool]],
                                    sample_shape: npt.NDArray[int]):
        curr_anomaly_shape = np.array(curr_anomaly_mask.shape)
        min_corner = np.zeros(len(sample_shape))
        max_corner = sample_shape - curr_anomaly_shape
        # - Apply anomaly at location
        while True:
            anomaly_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

            # If the sample mask is None, any location within the bounds is valid
            if sample_mask is None:
                break

            # Otherwise, we need to check that the intersection of the anomaly mask and the sample mask is at least 50%
            target_patch_obj_mask = sample_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)]
            if (np.sum(target_patch_obj_mask & curr_anomaly_mask) / np.sum(curr_anomaly_mask)) >= 0.5:
                break
        return anomaly_corner

    def __call__(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]], *args, **kwargs)\
            -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :param **kwargs:
            * *sample_path*: Path to source image
        :return: sample with task applied and label map.
        """
        return self.apply(sample, sample_mask, *args, **kwargs)

    @abstractmethod
    def augment_sample(self, sample: npt.NDArray[float], sample_mask: Optional[npt.NDArray[bool]],
                       anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool],
                       anomaly_intersect_fn: Callable[[npt.NDArray[float], npt.NDArray[float]], npt.NDArray[float]]) \
            -> npt.NDArray[float]:
        """
        Apply self-supervised task to region at anomaly_corner covered by anomaly_mask
        :param sample: Sample to be augmented.
        :param sample_mask: Object mask of sample.
        :param anomaly_corner: Index of anomaly corner.
        :param anomaly_mask: Mask
        :param anomaly_intersect_fn: Function which, given a line's origin and direction, finds its intersection with
        the edge of the anomaly mask
        :return:
        """
