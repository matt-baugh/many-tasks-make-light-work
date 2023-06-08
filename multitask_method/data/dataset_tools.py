from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class DatasetContainer(ABC):

    def __init__(self, sample_ids: List[str], load_data: bool):
        self.sample_ids = sample_ids
        self.load_data = load_data
        self.loaded_data = [self.load_sample(s) for s in tqdm(sample_ids)] if load_data else None

    @abstractmethod
    def load_sample(self, sample_id: str) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        pass

    def __getitem__(self, item) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]], str]:
        sample_id = self.sample_ids[item]
        sample, mask = self.loaded_data[item] if self.load_data else self.load_sample(sample_id)
        return sample, mask, sample_id

    def __len__(self):
        return len(self.sample_ids)

    def get_random_sample(self) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        return self[np.random.randint(len(self))][:2]


class DatasetCoordinator(ABC):

    @abstractmethod
    def dataset_size(self) -> int:
        pass

    def __len__(self):
        return self.dataset_size()

    @abstractmethod
    def dataset_dimensions(self) -> int:
        pass

    @abstractmethod
    def make_container(self, sample_indices: List[int]) -> DatasetContainer:
        pass
