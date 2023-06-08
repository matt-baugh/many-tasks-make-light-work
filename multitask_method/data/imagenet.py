from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from multitask_method.data.dataset_tools import DatasetContainer, DatasetCoordinator


class ImagenetDatasetContainer(DatasetContainer):
    def __init__(self, sample_ids: List[str], image_folder: Path):
        self.image_folder = image_folder
        super().__init__(sample_ids, True)

    def load_sample(self, sample_id: str) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        # noinspection PyTypeChecker
        sample_arr = np.load(self.image_folder / f'{sample_id}.npy')
        assert len(sample_arr.shape) == 2, f'Expected 2D image, got shape {sample_arr.shape}'

        return sample_arr[None], None


class ImagenetDatasetCoordinator(DatasetCoordinator):
    def __init__(self, dataset_root: Path, fullres: bool):
        # Datasets must be stored in a folder with the following structure:
        # dataset_root
        #   lowres
        #       <sample_id>.npy
        #   fullres
        #       [same as lowres]

        self.dataset_root = dataset_root
        self.fullres = fullres

        self.dset_folder = dataset_root / ('fullres' if fullres else 'lowres')
        self.sample_ids = sorted([f.stem for f in self.dset_folder.iterdir() if f.is_file()])
        assert len(self.sample_ids) == 10000, 'Imagenet subset expect 10k samples, found ' + str(len(self.sample_ids))

    def make_container(self, sample_indices: List[int]) -> DatasetContainer:

        return ImagenetDatasetContainer([self.sample_ids[i] for i in sample_indices], self.dset_folder)

    def dataset_size(self):
        return len(self.sample_ids)

    def dataset_dimensions(self) -> int:
        return 2
