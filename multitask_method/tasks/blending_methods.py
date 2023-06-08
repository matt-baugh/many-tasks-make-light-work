import numpy as np
import numpy.typing as npt

from pietorch import blend_dst_numpy

from multitask_method.tasks.utils import get_patch_image_slices


def cut_paste(sample: npt.NDArray[float], source_to_blend: npt.NDArray[float], anomaly_corner: npt.NDArray[int],
              anomaly_mask: npt.NDArray[bool]) -> npt.NDArray[float]:
    num_channels = sample.shape[0]
    repeated_mask = np.broadcast_to(anomaly_mask, source_to_blend.shape)
    sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))

    aug_sample = sample.copy()
    aug_sample[sample_slices][repeated_mask] = source_to_blend[repeated_mask]

    return aug_sample


def patch_interpolation(sample: npt.NDArray[float], source_to_blend: npt.NDArray[float],
                        anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool]) -> npt.NDArray[float]:

    num_channels = sample.shape[0]
    repeated_mask = np.broadcast_to(anomaly_mask, source_to_blend.shape)
    sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))

    factor = np.random.uniform(0.05, 1.0)

    aug_sample = sample.copy()
    aug_sample[sample_slices][repeated_mask] = (1 - factor) * aug_sample[sample_slices][repeated_mask] +\
        factor * source_to_blend[repeated_mask]

    return sample


def poisson_image_editing(sample: npt.NDArray[float], source_to_blend: npt.NDArray[float],
                          anomaly_corner: npt.NDArray[int], anomaly_mask: npt.NDArray[bool], mix_gradients: bool) \
        -> npt.NDArray[float]:
    raw_blended = blend_dst_numpy(sample, source_to_blend, anomaly_mask, anomaly_corner, mix_gradients, channels_dim=0)
    spatial_axis = tuple(range(1, len(sample.shape)))
    return np.clip(raw_blended, sample.min(axis=spatial_axis, keepdims=True), sample.max(axis=spatial_axis, keepdims=True))
