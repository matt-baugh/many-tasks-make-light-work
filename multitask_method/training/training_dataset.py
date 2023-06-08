from collections import namedtuple
from typing import List, Optional, Callable, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from multitask_method.pos_encoding import PosEnc
from multitask_method.data.dataset_tools import DatasetContainer
from multitask_method.tasks.base_task import BaseTask

TrainDatasetItem = namedtuple('TrainDatasetItem', ['normal_img', 'normal_pixel_label', 'normal_sample_label',
                                                   'aug_img', 'aug_pixel_label', 'aug_sample_label',
                                                   'pos_enc'])


class TrainingDataset(Dataset):
    def __init__(self, dset_container: DatasetContainer, tasks: List[BaseTask], pos_encoding: Optional[PosEnc],
                 train_transforms: Optional[Callable[[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]]],
                 dset_size_cap: Optional[int]):
        self.dset_container = dset_container
        assert len(tasks) > 0, 'Cannot train with 0 tasks'
        self.tasks = tasks
        self.pos_encoding = pos_encoding
        self.dset_size_cap = dset_size_cap
        self.train_transforms = train_transforms
        self.dset_subset = []

    def __len__(self):
        return len(self.dset_container)

    def __getitem__(self, index: int) -> TrainDatasetItem:

        if self.dset_size_cap is not None:
            # If we're only using a subset of the dataset
            if len(self.dset_subset) < self.dset_size_cap:
                curr_sample = self.dset_container[index]
                self.dset_subset.append(curr_sample)
            else:
                curr_sample = self.dset_subset[index % self.dset_size_cap]
        else:
            # Otherwise, just use dataset container normally
            curr_sample = self.dset_container[index]

        sample, mask, file_id = curr_sample

        if self.train_transforms is not None:
            sample, mask = self.train_transforms(sample, mask)

        aug_sample, pixel_label = np.random.choice(self.tasks)(sample, mask, sample_path=file_id)

        torch_normal_sample = torch.from_numpy(sample.copy())
        torch_aug_sample = torch.from_numpy(aug_sample)

        # check in labeller to avoid different shapes
        if len(pixel_label.shape) != len(aug_sample.shape):
            pixel_label = np.expand_dims(pixel_label, 0)

        torch_aug_pixel_label = torch.from_numpy(pixel_label)
        torch_normal_pixel_label = torch.zeros_like(torch_aug_pixel_label)

        return TrainDatasetItem(normal_img=torch_normal_sample, normal_pixel_label=torch_normal_pixel_label,
                                normal_sample_label=0.0,
                                aug_img=torch_aug_sample, aug_pixel_label=torch_aug_pixel_label, aug_sample_label=1.,
                                pos_enc=self.pos_encoding(sample.shape[1:]) if self.pos_encoding is not None else 0.)
