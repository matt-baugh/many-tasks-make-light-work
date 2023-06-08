import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import reconstruction

from multitask_method.data.dataset_tools import DatasetContainer
from multitask_method.plotting_utils import display_cross_section, display_normalised_cross_section
from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.patch_blending_task import PoissonImageEditingMixedGradBlender
from multitask_method.tasks.deformation_task import BendSinkDeformationTask, BendSourceDeformationTask
from multitask_method.tasks.labelling import IntensityDiffLabeller
from multitask_method.training.train_setup import construct_datasets, NUM_TASKS

from test.example_configs import example_hcp_config, example_vindr_config


def smoothing_visualisation(main_task: BaseTask, curr_labeller: IntensityDiffLabeller,
                            dset_container: DatasetContainer):

    example_idx = np.random.randint(len(dset_container))
    example_img, example_mask, _ = dset_container[example_idx]
    num_dims = len(example_img.shape[1:])

    aug_image, aug_image_binary_label = main_task(example_img, example_mask)
    direct_label = curr_labeller.label_fn(np.mean(aug_image_binary_label * np.abs(aug_image - example_img), axis=0))

    neighbour_footprint = ndimage.generate_binary_structure(num_dims, 1)

    extended_neighbour_footprint = ndimage.iterate_structure(neighbour_footprint, 2)

    def recon_helper(mask_img):
        recon_seed_img = np.copy(mask_img)
        recon_seed_img[num_dims * (slice(1, -1),)] = mask_img.max()
        return reconstruction(recon_seed_img, mask_img, method='erosion', footprint=neighbour_footprint)

    base_images = [
        ('Image', aug_image[0]),
        ('Binary label', aug_image_binary_label[0]),
        ('Direct label', direct_label),
        ('Direct Morph recon', recon_helper(direct_label)),
        ('Closed Morph recon', recon_helper(ndimage.grey_closing(direct_label, footprint=neighbour_footprint))),
        ('Closed Morph recon 1.5r',
         recon_helper(ndimage.grey_closing(direct_label, footprint=ndimage.generate_binary_structure(num_dims, 2))))
    ]

    def make_sphere_mask(r):
        diam_range = np.arange(-r, r + 1)
        mg = np.meshgrid(*([diam_range] * num_dims))
        return np.sum([D ** 2 for D in mg], axis=0) <= r ** 2

    closing_shapes = [
        ('Direct neighbours', neighbour_footprint),
        ('1.5 radius', ndimage.generate_binary_structure(num_dims, 2)),
        ('3 kernel', np.ones(num_dims * (3,), dtype=bool)),
        ('2 manhattan', extended_neighbour_footprint),
        ('3 manhattan', ndimage.binary_dilation(extended_neighbour_footprint)),
        ('2 radius', make_sphere_mask(2)),
        ('5 Kernel', np.ones(num_dims * (5,), dtype=bool))
    ]

    closing_images = [(c_name, ndimage.grey_closing(direct_label, footprint=c)) for c_name, c in closing_shapes]

    if num_dims == 3:
        ax_row_params = {'fontsize': 30, 'fontname': 'Times New Roman', 'labelpad': 15}
        num_base_rows = len(base_images)
        num_rows = num_base_rows + len(closing_shapes)

        anom_centre = np.round(ndimage.center_of_mass(direct_label, labels=ndimage.label(aug_image_binary_label)[0],
                                                      index=1)).astype(int)

        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))

        for a, (img_name, img_to_show) in zip(axes[:num_base_rows], base_images):

            if img_to_show.dtype == float:
                display_normalised_cross_section(img_to_show, *anom_centre, existing_fig_ax=(fig, a))
            else:
                display_cross_section(img_to_show, *anom_centre, existing_fig_ax=(fig, a))

            a[0].set_ylabel(img_name, **ax_row_params)

        for a, (c_name, c_img) in zip(axes[num_base_rows:], closing_images):
            display_cross_section(c_img, *anom_centre, existing_fig_ax=(fig, a))
            a[0].set_ylabel(c_name, **ax_row_params)
    elif num_dims == 2:
        ax_title_params = {'fontsize': 30, 'fontname': 'Times New Roman', 'pad': 15}

        all_images = base_images + closing_images
        num_images = len(all_images)

        images_per_row = 3
        num_rows = int(np.ceil(num_images / images_per_row))
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(18, 6 * num_rows))

        for a, (img_name, img_to_show) in zip(axes.flatten(), all_images):
            if img_name == 'Image':
                vmin, vmax = img_to_show.min(), img_to_show.max()
            else:
                vmin, vmax = 0., 1.
            a.imshow(img_to_show, vmin=vmin, vmax=vmax, cmap='gray')
            a.set_title(img_name, **ax_title_params)

    else:
        raise NotImplementedError

    fig.tight_layout()
    plt.show()


def visualise_patch_blending_labelling(exp_config):

    dset_container = exp_config.curr_dset_coord.make_container(list(range(10)))
    task = PoissonImageEditingMixedGradBlender(None, dset_container.get_random_sample, **exp_config.task_kwargs)

    smoothing_visualisation(task, exp_config.labeller, dset_container)


def test_blend_brain_labelling():
    visualise_patch_blending_labelling(example_hcp_config)


def test_blend_cxr_labelling():
    visualise_patch_blending_labelling(example_vindr_config)


def visualise_bend_source_labelling(exp_config):

    dset_container = exp_config.curr_dset_coord.make_container(list(range(10)))
    task = BendSourceDeformationTask(None, **exp_config.task_kwargs)

    smoothing_visualisation(task, exp_config.labeller, dset_container)


def visualise_bend_sink_labelling(exp_config):

    dset_container = exp_config.curr_dset_coord.make_container(list(range(10)))
    task = BendSinkDeformationTask(None, **exp_config.task_kwargs)

    smoothing_visualisation(task, exp_config.labeller, dset_container)


def test_source_brain_labelling():
    visualise_bend_source_labelling(example_hcp_config)


def test_source_cxr_labelling():
    visualise_bend_source_labelling(example_vindr_config)


def measure_label_speed(exp_config):
    # Run in profiler

    num_samples = 200
    num_dims = exp_config.curr_dset_coord.dataset_dimensions()

    dset_container = exp_config.curr_dset_coord.make_container(list(range(num_samples)))
    task = PoissonImageEditingMixedGradBlender(None, dset_container.get_random_sample, **exp_config.task_kwargs)

    labeller: IntensityDiffLabeller = exp_config.labeller

    neighbour_footprint = ndimage.generate_binary_structure(num_dims, 1)

    def recon_helper(mask_img):
        recon_seed_img = np.copy(mask_img)
        recon_seed_img[num_dims * (slice(1, -1),)] = mask_img.max()
        return reconstruction(recon_seed_img, mask_img, method='erosion', footprint=neighbour_footprint)

    for i in range(num_samples):
        example_img, example_mask, _ = dset_container[i]
        aug_image, aug_image_binary_label = task(example_img, example_mask)

        direct_label = labeller.label_fn(np.mean(aug_image_binary_label * np.abs(aug_image - example_img), axis=0))

        for anom_slice in ndimage.find_objects(ndimage.label(aug_image_binary_label[0])[0]):

            direct_label[anom_slice] = recon_helper(ndimage.grey_closing(direct_label[anom_slice],
                                                                         footprint=neighbour_footprint))


def test_hcp_label_speed():
    measure_label_speed(example_hcp_config)


def test_cxr_label_speed():
    measure_label_speed(example_vindr_config)


def visualise_all_tasks(exp_config):

    num_dims = exp_config.curr_dset_coord.dataset_dimensions()

    examples_per_task = 3
    num_samples = examples_per_task * NUM_TASKS

    train_dset, val_dset = construct_datasets(exp_config.curr_dset_coord, exp_config.other_dset_coord, 0, None,
                                              exp_config.num_train_tasks, exp_config.task_kwargs,
                                              exp_config.train_transforms, num_samples, exp_config.labeller)

    all_tasks = train_dset.tasks + val_dset.tasks

    if num_dims == 3:
        n_rows = num_samples * 2
    elif num_dims == 2:
        n_rows = num_samples
    else:
        raise NotImplementedError

    fig, axes = plt.subplots(n_rows, num_dims, figsize=(num_dims * 5, n_rows * 5))

    for task_index in range(NUM_TASKS):
        task = all_tasks[task_index]

        train_dset.tasks = [task]

        for example_index in range(examples_per_task):
            _, _, _, aug_image, aug_image_label, _, _ = train_dset[example_index]
            aug_image = aug_image[0].numpy()
            aug_image_label = aug_image_label[0].numpy()

            if num_dims == 3:
                anom_centre = np.round(
                    ndimage.center_of_mass(aug_image_label, labels=ndimage.label(aug_image_label > 0)[0],
                                           index=1)).astype(int)

                row_index = task_index * examples_per_task * 2 + example_index * 2
                display_normalised_cross_section(aug_image, *anom_centre, existing_fig_ax=(fig, axes[row_index]))
                display_cross_section(aug_image_label, *anom_centre, existing_fig_ax=(fig, axes[row_index + 1]))

                axes[row_index, 0].set_title(f"Task {task_index} - Example {example_index} - Image")
                axes[row_index + 1, 0].set_title(f"Task {task_index} - Example {example_index} - Label")

            elif num_dims == 2:
                row_index = task_index * examples_per_task + example_index
                axes[row_index, 0].imshow(aug_image, cmap='gray', vmin=aug_image.min(), vmax=aug_image.max())
                axes[row_index, 1].imshow(aug_image_label, cmap='gray', vmin=0, vmax=1)

                axes[row_index, 0].set_title(f"Task {task_index} - Example {example_index}")
            else:
                raise NotImplementedError

    fig.tight_layout()
    plt.show()


def test_visualise_all_hcp_tasks():
    visualise_all_tasks(example_hcp_config)


def test_visualise_all_cxr_tasks():
    visualise_all_tasks(example_vindr_config)



