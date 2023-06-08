import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.paths import base_prediction_dir
from multitask_method.plotting_utils import display_normalised_cross_section, display_cross_section
from multitask_method.utils import make_exp_config

NUM_EXAMPLES = 5


def main(exp_config, fold_name):

    exp_name = exp_config.name

    test_dset_name: str
    test_dset_coordinator: DatasetCoordinator

    ax_title_params = {'fontsize': 30, 'fontname': 'Times New Roman', 'pad': 15}
    ax_row_params = {'fontsize': 30, 'fontname': 'Times New Roman', 'labelpad': 15}
    for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():

        print(f'Visualising predictions on on {test_dset_name}')

        dset_pred_dir = base_prediction_dir / exp_name / test_dset_name / fold_name
        assert dset_pred_dir.is_dir(), f'Missing predictions: {dset_pred_dir}'

        num_dims = test_dset_coordinator.dataset_dimensions()

        example_indices = np.random.default_rng(seed=42).choice(len(test_dset_coordinator), NUM_EXAMPLES, replace=False)
        test_dset_container = test_dset_coordinator.make_container(example_indices)

        if num_dims == 3:
            n_rows = NUM_EXAMPLES * 3
        elif num_dims == 2:
            n_rows = NUM_EXAMPLES
        else:
            raise NotImplementedError

        # Always 3 columns, for 3 views in 3D case, or for 3 files in 2D case
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), constrained_layout=True)

        for i in range(NUM_EXAMPLES):

            example_img, example_label, example_id = test_dset_container[i]

            max_label = max(example_label.max(), 1)

            curr_pred = np.load(dset_pred_dir / f'{example_id}.npy')

            curr_row = i * 3 if num_dims == 3 else i

            if num_dims == 3:

                bin_label = (example_label > 0).astype(int)
                anom_instance_map = ndimage.label(bin_label)[0]

                labels, label_counts = np.unique(anom_instance_map[anom_instance_map != 0], return_counts=True)
                biggest_label = labels[np.argmax(label_counts)]

                anom_centre = np.round(ndimage.center_of_mass(bin_label, anom_instance_map, index=biggest_label)).astype(int)

                display_normalised_cross_section(example_img[0], *anom_centre, existing_fig_ax=(fig, axes[curr_row]))
                axes[curr_row][0].set_ylabel('Input', **ax_row_params)

                display_cross_section(example_label / max_label, *anom_centre, existing_fig_ax=(fig, axes[curr_row + 1]))
                axes[curr_row + 1][0].set_ylabel('Label', **ax_row_params)

                display_cross_section(curr_pred, *anom_centre, existing_fig_ax=(fig, axes[curr_row + 2]))
                axes[curr_row + 2][0].set_ylabel('Prediction', **ax_row_params)
            elif num_dims == 2:
                axes[i, 0].imshow(example_img[0], cmap='gray', vmin=example_img.min(), vmax=example_img.max())
                axes[i, 1].imshow(example_label / max_label, cmap='gray', vmin=0, vmax=1)
                axes[i, 2].imshow(curr_pred, cmap='gray', vmin=0, vmax=1)

        if num_dims == 2:
            axes[0, 0].set_title('Input', **ax_title_params)
            axes[0, 1].set_title('Label', **ax_title_params)
            axes[0, 2].set_title('Prediction', **ax_title_params)

        fig.suptitle(test_dset_name, fontsize=30, fontname='Times New Roman')
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict for test sets of a given experiment")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("FOLD", type=str, help="Fold to visualise (or 'ensemble')")
    parser_args = parser.parse_args()

    main(make_exp_config(parser_args.EXP_PATH), parser_args.FOLD)
