import argparse
import json

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.data.preprocessed_mri import PreprocessedMRIDatasetContainer
from multitask_method.paths import base_prediction_dir
from multitask_method.preprocessing.brain_preproc import CROP_INFO_DIR, brats_samples, isles_samples
from multitask_method.utils import make_exp_config
from cradl_eval import reverse_mm_crop

fig_names = {
    'BRATS': 'BraTS',
    'DDAD_TEST': '$\mathregular{DDAD}_{\mathregular{ts}}$',
    'VINDR_TEST': '$\mathregular{VinDr}_{\mathregular{ts}}$'
}


def main(exp_config, fold_name, num_examples_per_row, num_rows, skip_first_row):
    exp_name = exp_config.name

    n_cols = num_examples_per_row
    n_rows = num_rows
    total_num_examples = n_cols * n_rows
    examples_per_dset = total_num_examples / len(exp_config.test_dsets)
    dset_examples_per_row = int(examples_per_dset // n_rows)
    assert examples_per_dset == round(examples_per_dset), 'Must have equal number of examples per dset'
    examples_per_dset = int(examples_per_dset)

    test_dset_name: str
    test_dset_coordinator: DatasetCoordinator
    test_dset_containers = []
    for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():
        rng = np.random.default_rng(seed=1234)
        first_row_indices = rng.choice(len(test_dset_coordinator), dset_examples_per_row, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(test_dset_coordinator)), first_row_indices)
        if skip_first_row:
            example_indices = rng.choice(remaining_indices, examples_per_dset, replace=False)
        else:
            example_indices = np.concatenate([first_row_indices,
                                              rng.choice(remaining_indices,
                                                         examples_per_dset - dset_examples_per_row,
                                                         replace=False)])

        print(f'Visualising predictions on on {test_dset_name} (examples {example_indices})')
        test_dset_containers.append((test_dset_name, test_dset_coordinator.make_container(example_indices)))

    tmp_example_label = test_dset_containers[0][1][0][1]
    num_dims = len(tmp_example_label.shape)
    aspect_ratio = tmp_example_label.shape[-2] / tmp_example_label.shape[-1]

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5 + 1), layout='constrained')
    dset_subfigs = fig.subfigures(1, len(exp_config.test_dsets))

    for (test_dset_name, test_dset_container), test_dset_subfig in zip(sorted(test_dset_containers), dset_subfigs):

        dset_pred_dir = base_prediction_dir / exp_name / test_dset_name / fold_name
        assert dset_pred_dir.is_dir(), f'Missing predictions: {dset_pred_dir}'

        test_dset_axes = test_dset_subfig.subplots(n_rows, dset_examples_per_row)

        if n_rows == 1:
            test_dset_axes = [test_dset_axes]

        for i, row_axes in enumerate(test_dset_axes):
            for j, img_ax in enumerate(row_axes):

                example_img, example_label, example_id = test_dset_container[i * dset_examples_per_row + j]

                curr_pred = np.load(dset_pred_dir / f'{example_id}.npy')

                example_id = example_id.split('.')[0]

                if num_dims == 3:
                    test_dset_container: PreprocessedMRIDatasetContainer

                    # Get raw sample for brain visualisations
                    crop_json_path = test_dset_container.image_folder.parent / CROP_INFO_DIR / f'{example_id}.json'
                    with open(crop_json_path, 'r') as f:
                        crop_info = json.load(f)

                    curr_pred = reverse_mm_crop(curr_pred, crop_info, test_dset_name == 'BRATS')

                    sample_path_dict = dict(brats_samples if test_dset_name == 'BRATS' else isles_samples)
                    _, t2_path, seg_path = sample_path_dict[example_id]

                    example_img = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))[None]
                    example_label = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

                    bin_label = (example_label > 0).astype(int)
                    anom_instance_map = ndimage.label(bin_label)[0]

                    labels, label_counts = np.unique(anom_instance_map[anom_instance_map != 0], return_counts=True)
                    biggest_label = labels[np.argmax(label_counts)]

                    centre_z, _, _ = ndimage.center_of_mass(bin_label, anom_instance_map, index=biggest_label)
                    centre_z = int(centre_z)

                    img_to_show = example_img[0, centre_z]
                    label_to_show = example_label[centre_z]
                    pred_to_show = curr_pred[centre_z]
                elif num_dims == 2:
                    img_to_show = example_img[0]
                    label_to_show = example_label
                    pred_to_show = curr_pred
                else:
                    raise NotImplementedError

                img_ax.imshow(img_to_show, cmap='gray', vmin=example_img[0].min(), vmax=example_img[0].max())

                img_ax.imshow(pred_to_show, alpha=0.5, vmin=0, vmax=1)
                if not np.array_equal(np.unique(label_to_show), [0.]):
                    label_to_show = label_to_show != 0
                    label_contours = measure.find_contours(label_to_show)
                    for contour in label_contours:
                        img_ax.plot(contour[:, 1], contour[:, 0], color='r')

        for ax_row in test_dset_axes:
            for a in ax_row:
                a.set_xticks([])
                a.set_yticks([])

        test_dset_subfig.suptitle(fig_names.get(test_dset_name, test_dset_name), fontsize=70,
                                  fontname='Times New Roman')

    fig.savefig(f'EXTRA_paper_figure_{exp_name}_{fold_name}.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict for test sets of a given experiment")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("FOLD", type=str, help="Fold to visualise (or 'ensemble')")
    parser.add_argument("--NUM_EXAMPLES", type=int, default=8, help="Number of examples per row")
    parser.add_argument("--num_rows", type=int, default=1, help="Number of rows of examples", required=False)
    parser.add_argument("--skip_first", action='store_true', help="Skip first row of examples")
    parser_args = parser.parse_args()

    main(make_exp_config(parser_args.EXP_PATH), parser_args.FOLD, parser_args.NUM_EXAMPLES, parser_args.num_rows,
         parser_args.skip_first)
