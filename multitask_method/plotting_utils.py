import numpy as np
import matplotlib.pyplot as plt
import scipy

from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.utils import get_patch_slices
from multitask_method.training.train_setup import construct_datasets


def display_cross_section(image: np.ndarray, z: int = None, y: int = None, x: int = None, existing_fig_ax=None,
                          add_colorbar=False):
    if x is None:
        x = np.floor(image.shape[2] / 2).astype(int)
    if y is None:
        y = np.floor(image.shape[1] / 2).astype(int)
    if z is None:
        z = np.floor(image.shape[0] / 2).astype(int)

    if existing_fig_ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, ax = existing_fig_ax
        assert len(ax) == 3, f'Must have 3 axes to show crossection: {ax}'

    imshow_ax0 = ax[0].imshow(image[z, :, :], cmap='gray', vmin=0, vmax=1)
    imshow_ax1 = ax[1].imshow(image[:, y, :], origin='lower', cmap='gray', vmin=0, vmax=1)
    imshow_ax2 = ax[2].imshow(image[:, :, x], origin='lower', cmap='gray', vmin=0, vmax=1)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    if add_colorbar:
        fig.colorbar(imshow_ax2, ax=ax, shrink=0.6)

    return imshow_ax0, imshow_ax1, imshow_ax2


def display_normalised_cross_section(image: np.ndarray, z: int = None, y: int = None, x: int = None,
                                     existing_fig_ax=None, add_colorbar=False):
    mask = scipy.ndimage.binary_fill_holes(image != 0)

    image = image.copy()
    image[mask] -= image.min()
    image[mask] /= image.max()

    return display_cross_section(image, z, y, x, existing_fig_ax, add_colorbar)


def augment_sample_and_label(augment_args, task: BaseTask):
    sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn = augment_args
    aug_sample = task.augment_sample(sample.copy(), sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn)

    anomaly_mask = np.zeros(sample.shape[1:], dtype=bool)
    anomaly_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)] |= curr_anomaly_mask

    return aug_sample, task.sample_labeller(aug_sample, sample, anomaly_mask)


def load_samples_to_augment(num_examples, exp_config):
    # Load the dataset
    dataset, _ = construct_datasets(exp_config.curr_dset_coord, exp_config.other_dset_coord, 0,
                                    pos_enc=None, num_train_tasks=1, task_kwargs=exp_config.task_kwargs,
                                    dset_size_cap=num_examples, labeller=exp_config.labeller,
                                    train_transforms=exp_config.train_transforms)

    tmp_task = dataset.tasks[0]

    def prepare_sample_for_augment(sample, sample_mask, example_idx):
        sample_shape = np.array(sample.shape[1:])

        min_dim_lens = (0.06 * sample_shape).round().astype(int)
        max_dim_lens = (0.8 * sample_shape).round().astype(int)

        # To make sure we have a variety of shapes, limit the dimension according to the index
        # of the example
        dim_bounds_step = (max_dim_lens - min_dim_lens) / num_examples

        min_dim_lens += (dim_bounds_step * example_idx).round().astype(int)
        max_dim_lens = min_dim_lens + dim_bounds_step.round().astype(int)

        dim_bounds = list(zip(min_dim_lens, max_dim_lens))

        curr_anomaly_mask, intersect_fn = tmp_task.anomaly_shape_maker.get_patch_mask_and_intersect_fn(dim_bounds,
                                                                                                       sample_shape)
        # Choose anomaly location
        anomaly_corner = tmp_task.find_valid_anomaly_location(curr_anomaly_mask, sample_mask, sample_shape)

        return sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn

    return [prepare_sample_for_augment(*(dataset.dset_container[i][:2]), i) for i in range(num_examples)]
