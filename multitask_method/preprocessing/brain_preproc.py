import json
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage
import SimpleITK as sitk
from skimage import transform
from tqdm import tqdm

from multitask_method.paths import base_data_input_dir

# each bbox must be LEQ this (will be padded once loaded)
brain_input_size = np.array([160, 224, 160])


def z_norm(img: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    output = img.copy()

    mean = np.mean(img[mask])
    std = np.std(img[mask]) + 1e-8

    output[mask] = (img[mask] - mean) / std

    return output


def resize_preserve_edge(img_arr: npt.NDArray[float], new_size: npt.NDArray[int]):
    # Resampling the images prior to noramlisation blurs the boundaries
    # As the normalisation uses the non-zero pixels as a mask, the blurred boundaries add a lot of weight towards the
    # lower end of the distribution
    # We counteract this by using the resampled mask to correct the weighting differences

    img_mask = (img_arr > 0).astype(float)
    new_img = transform.resize(img_arr, new_size, order=1, preserve_range=True)
    new_mask = transform.resize(img_mask, new_size, order=1, preserve_range=True)

    bg_mask = new_mask < 0.5
    fg_mask = new_mask >= 0.5

    new_img[bg_mask] = 0
    new_img[fg_mask] = new_img[fg_mask] / new_mask[fg_mask]
    return new_img


def resize_brain(t1_arr: npt.NDArray[float], t2_arr: npt.NDArray[float], seg_arr: npt.NDArray[int],
                 new_size: npt.NDArray[int], maintain_edges: bool):
    if maintain_edges:
        t1_arr_resize = resize_preserve_edge(t1_arr, new_size)
        t2_arr_resize = resize_preserve_edge(t2_arr, new_size)
    else:
        t1_arr_resize = transform.resize(t1_arr, new_size, order=1, preserve_range=True)
        t2_arr_resize = transform.resize(t2_arr, new_size, order=1, preserve_range=True)

    seg_arr_resize = transform.resize(seg_arr, new_size, order=0, preserve_range=True, anti_aliasing=False)
    return t1_arr_resize, t2_arr_resize, seg_arr_resize


def load_and_crop(t1_path: Path, t2_path: Path, seg_path: Path, dset_name: str) -> Tuple[npt.NDArray, npt.NDArray, dict]:

    t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_path)).astype(float)
    t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path)).astype(float)
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    assert t1_arr.shape == t2_arr.shape, f'Shape mismatch: {t1_arr.shape}, {t2_arr.shape}'
    assert t1_arr.shape == seg_arr.shape, f'Shape mismatch: {t1_arr.shape}, {seg_arr.shape}'

    if 'hcp' in dset_name:
        # Correct spacing to be 1mm
        norm_shape = np.array(t1_arr.shape) * 0.7
        t1_arr, t2_arr, seg_arr = resize_brain(t1_arr, t2_arr, seg_arr, norm_shape, True)

    t1_mask = t1_arr > 0
    t2_mask = t2_arr > 0
    t1_arr = z_norm(t1_arr, t1_mask)
    t2_arr = z_norm(t2_arr, t2_mask)

    orig_shape = t2_arr.shape

    t1_box = [(s.start, s.stop) for s in ndimage.find_objects(t1_mask.astype(int))[0]]
    t2_box = [(s.start, s.stop) for s in ndimage.find_objects(t2_mask.astype(int))[0]]
    box_slice = tuple([slice(min(t1_min, t2_min), max(t1_max, t2_max))
                       for (t1_min, t1_max), (t2_min, t2_max) in zip(t1_box, t2_box)])

    orig_brain_bbox = [(s.start, s.stop) for s in box_slice]

    t1_arr = t1_arr[box_slice]
    t2_arr = t2_arr[box_slice]
    seg_arr = seg_arr[box_slice]
    full_res_shape = t2_arr.shape

    assert np.less_equal(t1_arr.shape, brain_input_size).all(), f'Bbox size: {t1_arr} too big!'

    if dset_name == 'brats17':
        t1_arr = t1_arr[:, ::-1, :]
        t2_arr = t2_arr[:, ::-1, :]
        seg_arr = seg_arr[:, ::-1, :]

    full_res = np.stack([t1_arr, t2_arr, seg_arr])

    t1_arr, t2_arr, seg_arr = [np.pad(c, tuple([(0, s % 2) for s in c.shape])) for c in [t1_arr, t2_arr, seg_arr]]

    low_res = np.stack(resize_brain(t1_arr, t2_arr, seg_arr, np.array(t1_arr.shape) / 2, False))
    low_res_brain_bbox = low_res.shape[1:]

    crop_info = {'orig_shape': orig_shape,
                 'orig_brain_bbox': orig_brain_bbox,
                 'full_res_shape': full_res_shape,
                 'low_res_shape': low_res_brain_bbox}

    return full_res, low_res, crop_info


# Constants
LOW_RES_DIR = 'lowres'
FULL_RES_DIR = 'fullres'
CROP_INFO_DIR = 'crop_info'

# HCP
hcp_raw_root = Path('/vol/biomedic3/mb4617/hcp/ood_hcp/')
hcp_sample_folders = [f for f in hcp_raw_root.iterdir() if f.name.isnumeric()]

hcp_t1_file = 'T1w_acpc_dc_restore_brain.nii.gz'
hcp_t2_file = 'T2w_acpc_dc_restore_brain.nii.gz'
hcp_seg_file = 'wmparc.nii.gz'

hcp_samples = [(f.name, (f / hcp_t1_file, f / hcp_t2_file, f / hcp_seg_file))
               for f in hcp_sample_folders]


# BRATS 2017
brats_raw_root = Path('/vol/vipdata/data/brain/brats/2017/original/Brats17TrainingData/')
brats_subdirs = ['LGG', 'HGG']
brats_sample_folders = [f for p_dir in brats_subdirs for f in (brats_raw_root / p_dir).iterdir()]

brats_samples = [(f.name, (f / f'{f.name}_t1.nii.gz', f / f'{f.name}_t2.nii.gz', f / f'{f.name}_seg.nii.gz'))
                 for f in brats_sample_folders]

# ISLES 2015
isles_raw_root = Path('/vol/vipdata/data/brain/ISLES_2015/SISS2015_Training/')


def isles_get_file(isles_sample_folder, modality_str):
    mod_subfolder = [f for f in isles_sample_folder.iterdir() if modality_str in f.name]
    assert len(mod_subfolder) == 1, (mod_subfolder, isles_sample_folder)
    mod_subfolder = mod_subfolder[0]
    return mod_subfolder / f'{mod_subfolder.name}.nii'


isles_samples = [(f.name, (isles_get_file(f, 'MR_T1'), isles_get_file(f, 'MR_T2'), isles_get_file(f, 'OT')))
                 for f in isles_raw_root.iterdir()]


def preprocess_all(output_dir: Path):

    for (dset_name, all_samples) in [('hcp', hcp_samples), ('brats17', brats_samples), ('isles2015', isles_samples)]:

        dset_output_dir = output_dir / dset_name
        dset_output_dir.mkdir(exist_ok=True)

        crop_info_dir = dset_output_dir / CROP_INFO_DIR
        crop_info_dir.mkdir()

        full_res_dir = dset_output_dir / FULL_RES_DIR
        full_res_dir.mkdir()
        
        low_res_dir = dset_output_dir / LOW_RES_DIR
        low_res_dir.mkdir()

        for s_name, s_files in tqdm(all_samples, f'Processing {dset_name}'):
            s_full_res_arr, s_low_res_arr, crop_info = load_and_crop(*s_files, dset_name)

            np.save(full_res_dir / s_name, s_full_res_arr)
            np.save(low_res_dir / s_name, s_low_res_arr)

            with open(crop_info_dir / f'{s_name}.json', 'w') as f:
                json.dump(crop_info, f)


if __name__ == '__main__':
    preprocess_all(base_data_input_dir)
