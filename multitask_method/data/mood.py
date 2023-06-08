from pathlib import Path
from typing import List, Optional, Tuple, Literal

import numpy.typing as npt

from multitask_method.data.dataset_tools import DatasetContainer, DatasetCoordinator
from multitask_method.utils import load_nii_gz


class MOODDatasetContainer(DatasetContainer):
    def __init__(self, sample_ids: List[str], image_folder: Path, mask_folder: Path, z_normalise: bool):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.z_normalise = z_normalise
        super().__init__(sample_ids, True)

    def load_sample(self, sample_id: str) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        img = load_nii_gz(self.image_folder, sample_id)
        mask = load_nii_gz(self.mask_folder, sample_id).astype(bool)
        if self.z_normalise:
            foreground = img[mask]
            img[mask] = (foreground - foreground.mean()) / (foreground.std() + 1e-8)
        return img[None], mask


class MOODDatasetCoordinator(DatasetCoordinator):
    def __init__(self, mood_root: Path, dataset: Literal['Brain', 'Abdomen'], fullres: bool, z_normalise: bool = False):
        self.mood_root = mood_root
        self.dataset = dataset
        self.fullres = fullres
        self.z_normalise = z_normalise

    def make_container(self, sample_indices: List[int]) -> DatasetContainer:
        dataset_root = self.mood_root / self.dataset
        dataset_name = self.dataset[:5].lower()

        sample_ids = [f'{i:05d}.nii.gz' for i in sample_indices]
        image_folder = dataset_root / f'{dataset_name}_train{"_lowres" if not self.fullres else ""}'
        mask_folder = dataset_root / f'{dataset_name}_mask{"_lowres" if not self.fullres else ""}'

        return MOODDatasetContainer(sample_ids, image_folder, mask_folder, self.z_normalise)

    def dataset_size(self):
        return 800 if self.dataset == 'Brain' else 550

    def dataset_dimensions(self) -> int:
        return 3
