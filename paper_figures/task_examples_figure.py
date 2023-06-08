import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.ndimage import binary_erosion
from skimage import measure

from multitask_method.plotting_utils import load_samples_to_augment, augment_sample_and_label
from multitask_method.tasks.deformation_task import RadialDeformationTask, BendSinkDeformationTask,\
    BendSourceDeformationTask
from multitask_method.tasks.patch_blending_task import TestPoissonImageEditingMixedGradBlender
from multitask_method.tasks.intensity_tasks import SmoothIntensityChangeTask

from test.example_configs import example_hcp_config, example_vindr_config


def plot_radial_task_examples(name, exp_config):

    num_examples = 5
    num_dims = exp_config.curr_dset_coord.dataset_dimensions()

    # Get samples to augment
    test_samples = load_samples_to_augment(num_examples, exp_config)

    # Get example from other dataset to blend in
    other_dset_sample, other_dset_mask, _ = exp_config.other_dset_coord.make_container([42])[0]

    task_kwargs = exp_config.task_kwargs
    task_kwargs['min_push_dist'] = 5
    task_kwargs['min_anom_prop'] = 0.15
    tasks = [
        ('Intra-dataset blending',
         TestPoissonImageEditingMixedGradBlender(exp_config.labeller, *test_samples[1][:2], **task_kwargs)),
        ('Inter-dataset blending',
         TestPoissonImageEditingMixedGradBlender(exp_config.labeller, other_dset_sample, other_dset_mask, **task_kwargs)),
        ('Sink deformation', BendSinkDeformationTask(exp_config.labeller, **task_kwargs)),
        ('Source deformation', BendSourceDeformationTask(exp_config.labeller, **task_kwargs)),
        ('Smooth intensity change', SmoothIntensityChangeTask(exp_config.labeller, **task_kwargs))
    ]
    assert len(tasks) == num_examples

    fig, ax = plt.subplots(2, num_examples, figsize=(5 * num_examples, 10))

    for i, ((task_name, curr_task), augment_args) in enumerate(zip(tasks, test_samples)):

        if isinstance(curr_task, RadialDeformationTask):
            inner_mask = binary_erosion(augment_args[3], iterations=5)
            def_centre = np.stack(np.nonzero(inner_mask))[:, np.random.randint(np.sum(inner_mask))]
            curr_task.deform_centre = def_centre
            anom_centre = augment_args[2] + def_centre
        else:
            anom_centre = augment_args[2] + np.array(augment_args[3].shape) // 2

        curr_aug_sample, curr_sample_label = augment_sample_and_label(augment_args, curr_task)

        if num_dims == 3:
            centre_z = anom_centre[0]
            img_to_show = curr_aug_sample[0, centre_z]
            healthy_img_to_show = augment_args[0][0, centre_z]
            label_to_show = curr_sample_label[centre_z]
        elif num_dims == 2:
            img_to_show = curr_aug_sample[0]
            healthy_img_to_show = augment_args[0][0]
            label_to_show = curr_sample_label
        else:
            raise NotImplementedError

        ax[0][i].set_title(task_name, fontsize=22)
        ax[0][i].imshow(img_to_show, cmap='gray', vmin=curr_aug_sample.min(), vmax=curr_aug_sample.max())
        ax[0][i].axis('off')

        label_contours = measure.find_contours(label_to_show > 0.01)
        for contour in label_contours:
            ax[0][i].plot(contour[:, 1], contour[:, 0], color='r', alpha=0.5)

        # Plot change in 1D slice
        full_aug_slice = img_to_show[anom_centre[-2]]
        full_healthy_slice = healthy_img_to_show[anom_centre[-2]]
        full_label_slice = label_to_show[anom_centre[-2]]
        anom_lb, anom_ub = np.nonzero(full_label_slice)[0][[0, -1]]

        plot_lb = max(0, anom_lb - 5)
        plot_ub = min(full_aug_slice.shape[0], anom_ub + 5)

        diff_coords = np.arange(anom_lb - 1, anom_ub + 2)

        ax[1][i].plot(diff_coords, full_healthy_slice[anom_lb - 1:anom_ub + 2], '--', color='grey')
        ax[1][i].plot(diff_coords, full_aug_slice[anom_lb - 1:anom_ub + 2], 'g')
        ax[1][i].plot(np.arange(plot_lb, anom_lb), full_healthy_slice[plot_lb:anom_lb], 'k')
        ax[1][i].plot(np.arange(anom_ub + 1, plot_ub), full_healthy_slice[anom_ub + 1:plot_ub], 'k')
        if isinstance(curr_task, RadialDeformationTask):
            ax[1][i].plot(anom_centre[-1], full_aug_slice[anom_centre[-1]], 'r*', markersize=10)

        # Add line linking the two using ConnectionPatch
        for x_coord in [anom_lb - 1, anom_ub + 1]:
            con = ConnectionPatch(xyA=(x_coord, anom_centre[-2]), xyB=(x_coord, full_healthy_slice[x_coord]),
                                  coordsA="data", coordsB="data",
                                  axesA=ax[0][i], axesB=ax[1][i], color="r", linestyle='--')
            fig.add_artist(con)

    fig.tight_layout()
    plt.show()
    # fig.savefig(f'{name}_task_example_fig.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot examples of tasks for paper')
    parser.add_argument("dataset", type=str, choices=['vindr', 'hcp'])

    args = parser.parse_args()
    dset = args.dataset

    if dset == 'vindr':
        plot_radial_task_examples(dset, example_vindr_config)
    elif dset == 'hcp':
        plot_radial_task_examples(dset, example_hcp_config)
    else:
        raise NotImplementedError
