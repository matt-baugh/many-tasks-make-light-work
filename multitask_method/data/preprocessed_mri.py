from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_fill_holes

from multitask_method.data.dataset_tools import DatasetContainer, DatasetCoordinator


class PreprocessedMRIDatasetContainer(DatasetContainer):
    def __init__(self, sample_ids: List[str], image_folder: Path, input_modalities: List[int],
                 target_size: npt.NDArray[int], foreground_mask: bool):
        self.image_folder = image_folder
        self.modalities = input_modalities
        self.target_size = target_size
        self.foreground_mask = foreground_mask
        super().__init__(sample_ids, True)

    def load_sample(self, sample_id: str) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        # Files are loaded as numpy arrays with channels as the first dimension
        # The channels are T1, T2, and segmentation (wm for HCP dataset, anomaly for other datasets)
        # Assumes that input modalities are z-normalised

        # noinspection PyTypeChecker
        sample_arr = np.load(self.image_folder / sample_id)

        # Pad with zeros evenly on both sides to target size
        pad_width = [(0, 0)] + [(int((t - s) / 2), int((t - s) / 2) + (t - s) % 2) for t, s in
                                zip(self.target_size, sample_arr.shape[1:])]
        sample_arr = np.pad(sample_arr, pad_width, 'constant')

        sample_output = sample_arr[self.modalities]

        sample_mask = binary_fill_holes((sample_output != 0).any(axis=0)) if self.foreground_mask else sample_arr[-1]

        return sample_output, sample_mask


class PreprocessedMRIDatasetCoordinator(DatasetCoordinator):
    def __init__(self, dataset_root: Path, input_modalities: List[int], fullres: bool, target_size: npt.NDArray[int],
                 foreground_mask: bool = True):
        # Datasets must be stored in a folder with the following structure:
        # dataset_root
        #   lowres
        #       <sample_id>.npy
        #   fullres
        #       <sample_id>.npy
        self.dataset_root = dataset_root
        self.input_modalities = input_modalities
        self.fullres = fullres
        self.target_size = target_size
        self.foreground_mask = foreground_mask

        self.sample_folder = dataset_root / ('fullres' if fullres else 'lowres')
        self.sample_ids = [f.name for f in self.sample_folder.iterdir() if f.is_file()]

    def make_container(self, sample_indices: List[int]) -> DatasetContainer:

        return PreprocessedMRIDatasetContainer([self.sample_ids[i] for i in sample_indices], self.sample_folder,
                                               self.input_modalities, self.target_size, self.foreground_mask)

    def dataset_size(self):
        return len(self.sample_ids)

    def dataset_dimensions(self) -> int:
        return 3
