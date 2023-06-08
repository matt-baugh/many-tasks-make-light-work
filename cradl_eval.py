import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from skimage import transform
import SimpleITK as sitk

from multitask_method.data.preprocessed_mri import PreprocessedMRIDatasetCoordinator
from multitask_method.preprocessing.brain_preproc import brats_samples, isles_samples, CROP_INFO_DIR, brain_input_size
from multitask_method.evaluation.eval_metrics import METRICS
from multitask_method.paths import base_prediction_dir
from multitask_method.utils import make_exp_config

CRADL_SIZE = 128
SLICE_OFFSET = 10


# This function performs both CRADL's preprocessing and test time augmentation
def cradl_crop(raw_img: npt.NDArray, is_brats: bool, is_label: bool) -> npt.NDArray:

    # Crop to middle 190, 155 in last two dimensions
    h, w = raw_img.shape[-2:]
    crop_img = raw_img[:, h // 2 - 95: h // 2 + 95, w // 2 - 77: w // 2 + 78]

    # If it's brats:
    # - Pad final dimension with 5 either side
    # Otherwise
    # - reverse y-axis
    if is_brats:
        crop_img = np.pad(crop_img, ((0, 0), (0, 0), (5, 5)), mode='constant')
    else:
        crop_img = crop_img[:, ::-1, :]

    # Drop first and last SLICE_OFFSET slices
    crop_img = crop_img[SLICE_OFFSET:-SLICE_OFFSET]

    # For each slice:
    # - Pad each slice to at least CRADL_SIZE + 5 (skip as we know it's 190x155)
    # - Resize to CRADL_SIZE + 1 (skimage.resize as float, threshold at 0.5, no anti-aliasing)
    # - Centre crop to CRADL_SIZE (as we know it's CRADL_SIZE + 1, just cut last row and column)
    result = np.zeros((crop_img.shape[0], CRADL_SIZE, CRADL_SIZE))
    for i, cropped_slice in enumerate(crop_img):
        if is_label:
            slice_resize = transform.resize(cropped_slice.astype(float), (CRADL_SIZE + 1, CRADL_SIZE + 1), order=1,
                                            mode='edge', clip=True, anti_aliasing=False)
            slice_resize = (slice_resize >= 0.5).astype(int)
        else:
            slice_resize = transform.resize(cropped_slice, (CRADL_SIZE + 1, CRADL_SIZE + 1), order=1, clip=True,
                                            anti_aliasing=False)
        result[i] = slice_resize[:-1, :-1]

    return result


def reverse_mm_crop(crop_img: npt.NDArray, crop_info: dict, is_brats: bool) -> npt.NDArray:

    orig_shape = np.array(crop_info['orig_shape'])
    orig_brain_bbox = crop_info['orig_brain_bbox']
    full_res_shape = np.array(crop_info['full_res_shape'])
    low_res_shape = np.array(crop_info['low_res_shape'])

    low_res_input_size = brain_input_size // 2
    assert np.array_equal(low_res_input_size, crop_img.shape)

    low_res_slices = [slice(int((t - s) / 2), -(int((t - s) / 2) + (t - s) % 2)) for t, s in
                      zip(low_res_input_size, low_res_shape)]
    low_res_bbox = crop_img[tuple(low_res_slices)]
    assert np.array_equal(low_res_bbox.shape, low_res_shape), f'{low_res_bbox.shape} != {low_res_shape}'

    full_res_bbox = transform.resize(low_res_bbox, low_res_shape * 2, order=3, clip=True, anti_aliasing=False)
    full_res_shape_diff = np.array(full_res_bbox.shape) - full_res_shape
    assert np.all(np.isin(np.unique(full_res_shape_diff), [0, 1])), f'full_res_shape_diff = {full_res_shape_diff}'

    full_res_bbox = full_res_bbox[tuple([slice(s) for s in full_res_shape])]
    assert np.array_equal(full_res_bbox.shape, full_res_shape), f'{full_res_bbox.shape} != {full_res_shape}'

    if is_brats:
        full_res_bbox = full_res_bbox[:, ::-1, :]

    pad_to_orig = [(bbox_lb, orig_s - bbox_ub) for orig_s, (bbox_lb, bbox_ub) in zip(orig_shape, orig_brain_bbox)]
    img_orig_res = np.pad(full_res_bbox, pad_to_orig, mode='constant')
    assert np.array_equal(img_orig_res.shape, orig_shape), f'{img_orig_res.shape} != {orig_shape}'

    return img_orig_res


def cradl_eval(exp_path):

    exp_config = make_exp_config(exp_path)
    print('Evaluating at same resolution and with same slices as CRADL')

    exp_name = exp_config.name

    test_dset_name: str
    test_dset_coordinator: PreprocessedMRIDatasetCoordinator
    for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():

        assert test_dset_name in ['ISLES', 'BRATS'], 'Only ISLES and BRATS supported for CRADL evaluation'
        is_brats = test_dset_name == 'BRATS'

        print(f'Evaluating on {test_dset_name}')

        raw_samples = brats_samples if is_brats else isles_samples
        cradl_labels = sorted([(s[0], cradl_crop(sitk.GetArrayFromImage(sitk.ReadImage(s[1][2])), is_brats, True))
                               for s in raw_samples],
                              key=lambda x: x[0])
        cradl_label_arrays = [l[1] for l in cradl_labels]

        crop_info_paths = [test_dset_coordinator.dataset_root / CROP_INFO_DIR / f'{s_name}.json'
                           for s_name, _ in cradl_labels]
        crop_info_dicts = [json.load(open(cip, 'r')) for cip in crop_info_paths]

        dset_pred_dir = base_prediction_dir / exp_name / test_dset_name

        # Make folders to save predictions
        pred_folders: List[Path] = [p_dir for p_dir in dset_pred_dir.iterdir() if p_dir.is_dir()]

        print('Found prediction folders: ', pred_folders)

        all_results = {}
        for p_dir in pred_folders:
            print('Computing metrics for predictions in ', p_dir)

            pred_paths = sorted([p_f for p_f in p_dir.iterdir()])
            assert len(cradl_labels) == len(pred_paths), 'Different number of predictions vs labels'
            assert all([p_f.stem.split('.')[0] == l_f[0] for p_f, l_f in zip(pred_paths, cradl_labels)]), \
                'Mismatched prediction and label file names'

            # Load predictions, sorted by file name
            sample_predictions = [np.clip(np.squeeze(np.load(p_f)), 0, 1) for p_f in pred_paths]

            cradl_preds = [cradl_crop(reverse_mm_crop(pred, c_info, is_brats), is_brats, False)
                           for pred, c_info in zip(sample_predictions, crop_info_dicts)]

            all_results[p_dir.name] = {}
            for e_m in exp_config.eval_metrics:
                all_results[p_dir.name][e_m] = METRICS[e_m](cradl_preds, cradl_label_arrays, exp_config.eval_scales)

        print('Evaluation results:')
        print(json.dumps(all_results, indent=2))

        results_save_path = dset_pred_dir / 'cradl_results.json'
        print(f'Saving at ', results_save_path)

        with open(results_save_path, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate predictions on test sets of a given experiment")
    parser.add_argument("EXP_PATHS", type=str, nargs='+', help="Paths to experiment file")
    parser_args = parser.parse_args()

    with Pool(processes=min(4, len(parser_args.EXP_PATHS))) as p:

        async_results = [p.apply_async(cradl_eval, (exp_path,)) for exp_path in parser_args.EXP_PATHS]

        for async_result in async_results:
            async_result.wait()
            print(async_result.get())
