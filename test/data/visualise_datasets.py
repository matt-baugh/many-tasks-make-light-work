from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.paths import base_data_input_dir
from multitask_method.plotting_utils import display_normalised_cross_section, display_cross_section
from test.example_configs import example_hcp_config


def visualise_dataset(vis_name: str, dset_coord: DatasetCoordinator):
    num_dims = dset_coord.dataset_dimensions()

    num_samples = 5

    dset_container = dset_coord.make_container(list(range(num_samples)))

    if num_dims == 3:
        n_rows = num_samples * 2
    elif num_dims == 2:
        n_rows = num_samples
    else:
        raise NotImplementedError

    fig, axes = plt.subplots(n_rows, num_dims, figsize=(num_dims * 5, n_rows * 5))

    for i in range(num_samples):

        example_img, example_label, _ = dset_container[i]

        max_label = max(example_label.max(), 1)

        if num_dims == 3:
            # display_normalised_cross_section(example_img[0], existing_fig_ax=(fig, axes[2 * i]))
            display_cross_section(np.interp(example_img[0], [example_img[0].min(), example_img[0].max()], [0, 1]),
                                  existing_fig_ax=(fig, axes[2 * i]))
            display_cross_section(example_label / max_label, existing_fig_ax=(fig, axes[2 * i + 1]))
        elif num_dims == 2:
            axes[i, 0].imshow(example_img, cmap='gray', vmin=example_img.min(), vmax=example_img.max())
            axes[i, 1].imshow(example_label / max_label, cmap='gray', vmin=0, vmax=1)

    fig.suptitle(vis_name)
    fig.tight_layout()
    plt.show()


def test_visualise_hcp_curr_dset():
    visualise_dataset('Brain curr_dset', example_hcp_config.curr_dset_coord)


def test_visualise_hcp_other_dset():
    visualise_dataset('Brain other_dset', example_hcp_config.other_dset_coord)


def visualise_all_test_dsets(config_name: str, exp_config):
    for test_dset_name, test_dset_coord in exp_config.test_dsets.items():
        visualise_dataset(f'{config_name} test dset - {test_dset_name}', test_dset_coord)


def test_visualise_hcp_test_dsets():
    visualise_all_test_dsets('Brain', example_hcp_config)


def test_visualise_old_vs_new_hcp_train_dset():
    old_dset_coord = deepcopy(example_hcp_config.curr_dset_coord)
    old_dset_coord.sample_folder = base_data_input_dir / 'hcp' / 'lowres'
    visualise_dataset('Old HCP train', old_dset_coord)

    new_dset_coord = deepcopy(example_hcp_config.curr_dset_coord)
    new_dset_coord.sample_folder = base_data_input_dir / 'hcp' / 'lowres'
    visualise_dataset('New HCP train', new_dset_coord)
