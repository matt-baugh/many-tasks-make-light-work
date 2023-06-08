import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

from multitask_method.plotting_utils import display_cross_section, display_normalised_cross_section, \
    augment_sample_and_label, load_samples_to_augment
from multitask_method.tasks.deformation_task import SourceDeformationTask, SinkDeformationTask, \
    BendSinkDeformationTask, BendSourceDeformationTask

from test.example_configs import example_hcp_config, example_vindr_config


def plot_radial_task_examples(exp_config, task_class, varying_arg, varying_arg_vals, task_kwargs):
    num_examples = 3
    num_dims = exp_config.curr_dset_coord.dataset_dimensions()

    test_samples = load_samples_to_augment(num_examples, exp_config)
    test_deformation_centres = [np.stack(np.nonzero(t_s[3]))[:, np.random.randint(np.sum(t_s[3]))]
                                for t_s in test_samples]

    if num_dims == 2:
        fig, ax = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
    elif num_dims == 3:
        fig, ax = plt.subplots(num_examples * 2, 3, figsize=(15, 10 * num_examples))
    else:
        raise ValueError(f'Invalid number of dimensions: {num_dims}')

    all_imshow_axes = []

    for curr_var_arg_val in varying_arg_vals:

        curr_factor_axes = []

        for i, (augment_args, def_c) in enumerate(zip(test_samples, test_deformation_centres)):
            curr_task_kwargs = {varying_arg: curr_var_arg_val, 'deform_centre': def_c, **task_kwargs}
            curr_task = task_class(exp_config.labeller, **curr_task_kwargs)

            anom_centre = augment_args[2] + np.array(augment_args[3].shape) // 2

            curr_aug_sample, curr_sample_label = augment_sample_and_label(augment_args, curr_task)

            if num_dims == 2:
                curr_factor_axes.extend([ax[i][0].imshow(curr_aug_sample[0], cmap='gray',
                                                         vmin=curr_aug_sample[0].min(),
                                                         vmax=curr_aug_sample[0].max()),
                                         ax[i][1].imshow(curr_sample_label, cmap='gray', vmin=0, vmax=1)])
            elif num_dims == 3:
                anom_z, anom_y, anom_x = anom_centre
                curr_factor_axes.extend(
                    display_normalised_cross_section(curr_aug_sample[0], z=anom_z, y=anom_y, x=anom_x,
                                                     existing_fig_ax=(fig, ax[i * 2])))
                curr_factor_axes.extend(display_cross_section(curr_sample_label, z=anom_z, y=anom_y, x=anom_x,
                                                              existing_fig_ax=(fig, ax[i * 2 + 1])))
            else:
                raise ValueError(f'Invalid number of dimensions: {num_dims}')

        all_imshow_axes.append(curr_factor_axes)

    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, all_imshow_axes, interval=1000)
    ani.save(f'test_{exp_config.name}_{task_class.__name__}_visualisation.gif')


def test_brain_visualise_source_deformation():
    plot_radial_task_examples(example_hcp_config, SourceDeformationTask, 'deform_factor', np.arange(1, 4, 0.5), {})


def test_brain_visualise_sink_deformation():
    plot_radial_task_examples(example_hcp_config, SinkDeformationTask, 'deform_factor', np.arange(1, 4, 0.5), {})


def test_brain_visualise_bend_source_deformation():
    task_kwargs = {'min_push_dist': 0.5, 'max_push_dist': 5}
    plot_radial_task_examples(BendSourceDeformationTask, 'test_push_dist', np.arange(0, 5, 0.5), task_kwargs)


def test_brain_visualise_bend_sink_deformation():
    task_kwargs = {'min_push_dist': 0, 'max_push_dist': 5}
    plot_radial_task_examples(example_hcp_config, BendSinkDeformationTask, 'test_push_dist', np.arange(0, 5, 0.5),
                              task_kwargs)


def test_cxr_visualise_bend_source_deformation():
    task_kwargs = {'min_push_dist': 1, 'max_push_dist': 10}
    plot_radial_task_examples(example_vindr_config, BendSourceDeformationTask, 'test_push_dist', np.arange(0, 10, 0.5),
                              task_kwargs)


def test_cxr_visualise_bend_sink_deformation():
    task_kwargs = {'min_push_dist': 1, 'max_push_dist': 10}
    plot_radial_task_examples(example_vindr_config, BendSinkDeformationTask, 'test_push_dist', np.arange(0, 10, 0.5),
                              task_kwargs)
