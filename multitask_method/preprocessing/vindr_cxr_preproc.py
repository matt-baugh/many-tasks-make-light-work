from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Queue
import shutil
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
import pydicom
from skimage import exposure, transform
from tqdm import tqdm


from multitask_method.paths import base_data_input_dir

# Set output parameters
full_res = 512
low_res = 256
output_range = [-1., 1.]

raw_root = Path('/vol/biodata/data/chest_xray/VinDr-CXR/1.0.0_raw')
base_output_dir = base_data_input_dir / 'vindr_cxr'

# Debug - for disabling things during development
debug_save_files = True

# Preprocessing aims:
# - do initial resize before normalising
# - scaling with dicom Window Width/Center
# - histogram equalisation

ANNOTATIONS = 'annotations'
TRAIN = 'train'
TEST = 'test'

# Constants
FULL_RES_DIR = base_output_dir / 'fullres'
LOW_RES_DIR = base_output_dir / 'lowres'

TRAIN_IMAGE_LABELS = 'image_labels_train.csv'
TEST_IMAGE_LABELS = 'image_labels_test.csv'
TRAIN_ANNOTATIONS = 'annotations_train.csv'
TEST_ANNOTATIONS = 'annotations_test.csv'


def load_with_path(p: Path) -> Tuple[Path, pydicom.FileDataset]:
    return p, pydicom.dcmread(p)


class TmpDataLoader:

    def __init__(self, pool: ThreadPool, file_paths: List[Path]):
        self.pool = pool
        self.file_paths = file_paths
        self.prefetch = 10
        self.queue = Queue()

        self.index = 0
        self.last_added = False

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        self.index = 0
        self.last_added = False

        for _ in range(self.prefetch):
            self.queue_next_file()
        return self

    def queue_next_file(self):
        if self.index < len(self.file_paths):
            self.queue.put(self.pool.apply_async(load_with_path, (self.file_paths[self.index],)))
            self.index += 1
            if self.index == len(self.file_paths):
                self.last_added = True

    def __next__(self) -> Tuple[Path, pydicom.FileDataset]:
        if self.last_added and self.queue.empty():
            raise StopIteration

        curr_elem = self.queue.get()
        self.queue_next_file()

        return curr_elem.get()


def window_clip(dicom_img, np_img):
    if not hasattr(dicom_img, 'WindowCenter'):
        return np_img

    window_c = dicom_img.WindowCenter
    half_width = dicom_img.WindowWidth / 2
    w_min = window_c - half_width
    w_max = window_c + half_width
    return np.clip(np_img, w_min, w_max)


def preprocess_vindr_cxr(root_dset_path):

    annotations_dict, orig_image_labels, orig_test_dir, orig_train_dir = gen_vindr_structure(root_dset_path)

    # Sample-level labels can be directly copied across
    if debug_save_files:
        base_output_dir.mkdir(exist_ok=True)
        FULL_RES_DIR.mkdir(exist_ok=True)
        LOW_RES_DIR.mkdir(exist_ok=True)

        for img_lbl_file in orig_image_labels:
            for out_dir in [FULL_RES_DIR, LOW_RES_DIR]:
                shutil.copy(img_lbl_file, out_dir)


    global_pool = ThreadPool()

    for dir_type, dir_path in [(TRAIN, orig_train_dir), (TEST, orig_test_dir)]:
        print('Processing', dir_type, 'directory')
        bbox_csv = pd.read_csv(annotations_dict[dir_type])

        all_dicom_paths = [p for p in dir_path.iterdir() if p.suffix == '.dicom']
        data_loader = TmpDataLoader(global_pool, all_dicom_paths)

        fulL_res_out_dir = FULL_RES_DIR / dir_type
        low_res_out_dir = LOW_RES_DIR / dir_type

        if debug_save_files:
            fulL_res_out_dir.mkdir(exist_ok=True)
            low_res_out_dir.mkdir(exist_ok=True)

        for img_path, img_dicom in tqdm(data_loader, 'Processing images'):
            if img_path.suffix != '.dicom':
                continue

            image_id = img_path.stem
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                img_raw_np = img_dicom.pixel_array.astype(float)
            img_height, img_width = img_raw_np.shape

            img_bboxes = bbox_csv.image_id == image_id
            if dir_type == TRAIN:
                # All training images should be labelled by 3 radiologists, so should have at least 3 labels
                assert img_bboxes.sum() >= 3, f'Sample {image_id} has less than 3 annotations.'

            # Rescale bboxes for new image size
            bbox_csv.loc[img_bboxes, ['y_min', 'y_max']] *= (full_res / img_height)
            bbox_csv.loc[img_bboxes, ['x_min', 'x_max']] *= (full_res / img_width)

            full_res_raw_img = transform.resize(img_raw_np, output_shape=(full_res, full_res), order=3,
                                                anti_aliasing=True, preserve_range=True)

            full_res_norm_img = vindr_preproc_func(full_res_raw_img, img_dicom)

            full_res_centre_img = np.interp(full_res_norm_img, [np.min(full_res_norm_img), np.max(full_res_norm_img)],
                                            output_range)

            low_res_img = transform.resize(full_res_centre_img, output_shape=(low_res, low_res), order=3,
                                           anti_aliasing=True, preserve_range=True)

            if debug_save_files:

                global_pool.apply_async(np.save, (fulL_res_out_dir / img_path.stem, full_res_centre_img))
                global_pool.apply_async(np.save, (low_res_out_dir / img_path.stem, low_res_img))

        if debug_save_files:
            bbox_csv.to_csv(FULL_RES_DIR / annotations_dict[dir_type].name)

            bbox_csv.loc[:, ['y_min', 'y_max', 'x_min', 'x_max']] *= (low_res / full_res)
            bbox_csv.to_csv(LOW_RES_DIR / annotations_dict[dir_type].name)


def vindr_preproc_func(full_res_raw_img, img_dicom):
    full_res_norm_img = exposure.equalize_hist(window_clip(img_dicom, full_res_raw_img), 256)
    if img_dicom.PhotometricInterpretation == "MONOCHROME1":
        full_res_norm_img = full_res_norm_img.max() - full_res_norm_img
    return full_res_norm_img


def gen_vindr_structure(root_dset_path):
    orig_train_dir = root_dset_path / TRAIN
    orig_test_dir = root_dset_path / TEST
    orig_annotation_dir = root_dset_path / ANNOTATIONS
    orig_image_labels = [orig_annotation_dir / TRAIN_IMAGE_LABELS, orig_annotation_dir / TEST_IMAGE_LABELS]
    annotations_dict = {
        TRAIN: orig_annotation_dir / TRAIN_ANNOTATIONS,
        TEST: orig_annotation_dir / TEST_ANNOTATIONS
    }
    return annotations_dict, orig_image_labels, orig_test_dir, orig_train_dir


def generate_vindr_mask(sample_annotations, sample_arr):
    sample_mask = np.zeros_like(sample_arr, dtype=bool)
    # Get the annotations for this sample
    for _, row in sample_annotations.iterrows():
        if row['class_name'] != 'No finding':
            sample_mask[round(row['y_min']): round(row['y_max']), round(row['x_min']): round(row['x_max'])] = True
    return sample_mask


if __name__ == '__main__':
    preprocess_vindr_cxr(raw_root)

