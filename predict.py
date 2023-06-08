
import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import time
from typing import Dict

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.paths import base_prediction_dir, exp_log_dir, ensemble_dir
from multitask_method.training.train_setup import NUM_TASKS
from multitask_method.utils import make_exp_config


def get_all_sample_predictions(np_sample_img: npt.NDArray, fold_models: Dict[str, nn.Module], device: torch.device,
                               sample_pos_enc: torch.Tensor) -> Dict[str, npt.NDArray]:
    torch_sample_img = torch.from_numpy(np_sample_img).float().to(device)
    model_input = torch.unsqueeze(torch.cat([torch_sample_img, sample_pos_enc]), 0)

    curr_fold_preds: Dict[str, npt.NDArray] = {}
    model: nn.Module
    for fold, model in fold_models.items():
        assert not model.training
        with torch.no_grad():
            curr_fold_preds[fold] = model(model_input).squeeze().cpu().numpy()

    curr_fold_preds[ensemble_dir] = np.mean(list(curr_fold_preds.values()), axis=0)

    return curr_fold_preds


def main(exp_config, checkpoint_file):

    exp_name = exp_config.name
    num_folds = len(list(combinations(range(NUM_TASKS), exp_config.num_train_tasks)))
    device = torch.device("cuda")

    # Get models for each fold
    fold_models: Dict[str, nn.Module] = {}
    for f in tqdm(range(num_folds), 'Loading models'):
        fold = str(f)
        exp_fold_checkpoint_path = exp_log_dir(exp_name, fold) / checkpoint_file

        if exp_fold_checkpoint_path.is_file():
            exp_fold_checkpoint = torch.load(exp_fold_checkpoint_path)
            exp_fold_model = deepcopy(exp_config.model)
            exp_fold_model.load_state_dict(exp_fold_checkpoint["model_state_dict"])
            exp_fold_model.eval()
            exp_fold_model.to(device)

            fold_models[fold] = exp_fold_model
        else:
            print(f'WARNING: Missing model for fold {fold}, expected at {exp_fold_checkpoint_path}')

    num_folds_present = len(fold_models)
    if num_folds_present == 0:
        print('No models found, aborting prediction')
        exit()
    elif num_folds_present != num_folds:
        res = input(f'Continue with {num_folds_present} available models? [Y/n]').lower()
        if not (res == 'y' or res == ''):
            exit()

    test_dset_name: str
    test_dset_coordinator: DatasetCoordinator
    for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():

        print(f'Predicting on {test_dset_name}')
        dset_pred_start = time.time()

        dset_pred_dir = base_prediction_dir / exp_name / test_dset_name
        dset_pred_dir.mkdir(parents=True, exist_ok=True)

        # Make folders to save predictions
        pred_folders: Dict[str, Path] = {f: dset_pred_dir / f for f in fold_models.keys()}
        pred_folders[ensemble_dir] = dset_pred_dir / ensemble_dir
        for p_f in pred_folders.values():
            p_f.mkdir(exist_ok=True)

        print(f'Loading test samples for {test_dset_name}')
        test_dset_container = test_dset_coordinator.make_container(list(range(len(test_dset_coordinator))))

        # Get sample position encoding
        sample_pos_enc = torch.from_numpy(exp_config.pos_enc(test_dset_container[0][0].shape[1:])).float().to(device)

        for i in tqdm(range(len(test_dset_container)), f'Predicting test samples for {test_dset_name}'):
            np_sample_img, _, sample_id = test_dset_container[i]

            sample_preds = get_all_sample_predictions(np_sample_img, fold_models, device, sample_pos_enc)

            for fold, pred in sample_preds.items():
                np.save(pred_folders[fold] / f'{sample_id}.npy', pred)

        print(f'Finished predicting on {test_dset_name} in {time.time() - dset_pred_start} seconds')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict for test sets of a given experiment")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument('--checkpoint_file', default='best_model_val_loss.pt', type=str, help='Which checkpoint to use')
    parser_args = parser.parse_args()

    main(make_exp_config(parser_args.EXP_PATH), parser_args.checkpoint_file)
