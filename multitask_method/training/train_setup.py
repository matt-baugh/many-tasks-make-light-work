from typing import List, Literal, Tuple, Optional, Dict, Any, Callable
from itertools import combinations
import numpy as np
import numpy.typing as npt

from multitask_method.pos_encoding import PosEnc
from multitask_method.data.dataset_tools import DatasetCoordinator, DatasetContainer
from multitask_method.training.training_dataset import TrainingDataset
from multitask_method.tasks.base_task import BaseTask
from multitask_method.tasks.deformation_task import BendSinkDeformationTask, BendSourceDeformationTask
from multitask_method.tasks.intensity_tasks import SmoothIntensityChangeTask
from multitask_method.tasks.patch_blending_task import PoissonImageEditingMixedGradBlender
from multitask_method.tasks.labelling import AnomalyLabeller

NUM_TASKS = 5


def construct_datasets(curr_dset_coord: DatasetCoordinator, other_dset_coord: DatasetCoordinator, fold: int,
                       pos_enc: Optional[PosEnc], num_train_tasks: int, task_kwargs: Dict[str, Any],
                       train_transforms: Optional[Callable[[npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]]],
                       dset_size_cap: Optional[int] = None, labeller: Optional[AnomalyLabeller] = None) \
        -> Tuple[TrainingDataset, TrainingDataset]:
    assert 0 < num_train_tasks <= num_train_tasks, f'Invalid number of training tasks: {num_train_tasks}'
    assert curr_dset_coord.dataset_dimensions() == other_dset_coord.dataset_dimensions(), \
        'Dimensions of datasets must match!'

    # Select train/val task id's according to the current fold number
    all_task_ids = list(range(NUM_TASKS))
    train_task_ids = list(combinations(all_task_ids, num_train_tasks))[fold]
    val_task_ids = [t_id for t_id in all_task_ids if t_id not in train_task_ids]

    # Divide dataset samples into NUM_TASKS groups
    dataset_size = curr_dset_coord.dataset_size()
    all_sample_indices = list(range(dataset_size))
    sample_partitions = np.array_split(all_sample_indices, NUM_TASKS)

    train_sample_inds = [s_ind for i in train_task_ids for s_ind in sample_partitions[i]]
    val_sample_inds = [s_ind for i in val_task_ids for s_ind in sample_partitions[i]]

    # Always train on larger portion
    if len(val_sample_inds) > len(train_sample_inds):
        tmp_file_ids = train_sample_inds
        train_sample_inds = val_sample_inds
        val_sample_inds = tmp_file_ids

    if dset_size_cap is not None:
        train_sample_inds = train_sample_inds[:dset_size_cap]
        val_sample_inds = val_sample_inds[:dset_size_cap]

    # These are containers for main dataset
    train_dataset_container = curr_dset_coord.make_container(train_sample_inds)
    val_dataset_container = curr_dset_coord.make_container(val_sample_inds)

    # Load containers for other dataset (used in blending task)
    other_dset_size = other_dset_coord.dataset_size()
    # TODO: Is 50% of other dataset ok?
    other_dset_blend_inds = list(np.random.choice(list(range(other_dset_size)), other_dset_size // 2, replace=False))
    if dset_size_cap is not None:
        other_dset_blend_inds = other_dset_blend_inds[:dset_size_cap]
    other_dset_container = other_dset_coord.make_container(other_dset_blend_inds)

    def construct_task(t_id: int, curr_dataset_container: DatasetContainer) -> BaseTask:
        assert t_id < NUM_TASKS, f'Invalid task number: {t_id}'
        if t_id == 0:
            # Poisson blending within dataset
            print(' - Possion image blending within same dataset')
            return PoissonImageEditingMixedGradBlender(labeller, curr_dataset_container.get_random_sample, **task_kwargs)
        elif t_id == 1:
            print(' - Possion image blending from other dataset')
            # Poisson blending between datasets
            return PoissonImageEditingMixedGradBlender(labeller, other_dset_container.get_random_sample, **task_kwargs)
        elif t_id == 2:
            # Sink deformation
            print(' - Sink deformation')
            return BendSinkDeformationTask(labeller, **task_kwargs)
        elif t_id == 3:
            print(' - Source deformation')
            # Source deformation
            return BendSourceDeformationTask(labeller, **task_kwargs)
        elif t_id == 4:
            # Smooth intensity change
            print(' - Smooth intensity change')
            return SmoothIntensityChangeTask(labeller, **task_kwargs)
        assert False, 'How did you get here?'

    print('Training tasks:')
    train_tasks = [construct_task(task_id, train_dataset_container) for task_id in train_task_ids]
    print('\nValidation tasks:')
    val_tasks = [construct_task(task_id, val_dataset_container) for task_id in val_task_ids]

    train_dataset = TrainingDataset(train_dataset_container, train_tasks, pos_enc, train_transforms, dset_size_cap)
    val_dataset = TrainingDataset(val_dataset_container, val_tasks, pos_enc, train_transforms, dset_size_cap)

    return train_dataset, val_dataset
