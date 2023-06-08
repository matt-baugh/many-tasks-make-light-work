import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.evaluation.eval_metrics import METRICS, get_test_anom_data
from multitask_method.paths import base_prediction_dir
from multitask_method.utils import make_exp_config


def main(exp_config):

    exp_name = exp_config.name

    test_dset_name: str
    test_dset_coordinator: DatasetCoordinator
    for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():

        print(f'Evaluating on {test_dset_name}')

        dset_pred_dir = base_prediction_dir / exp_name / test_dset_name

        # Make folders to save predictions
        pred_folders: List[Path] = [p_dir for p_dir in dset_pred_dir.iterdir() if p_dir.is_dir()]

        print('Found prediction folders: ', pred_folders)

        print(f'Loading test labels for {test_dset_name}')
        all_test_data, sample_labels = get_test_anom_data(test_dset_coordinator)

        all_results = {}
        for p_dir in pred_folders:
            print('Computing metrics for predictions in ', p_dir)

            pred_paths = sorted([p_f for p_f in p_dir.iterdir()])
            assert len(sample_labels) == len(pred_paths), 'Different number of predictions vs labels'
            assert all([p_f.stem == l_f[2] for p_f, l_f in zip(pred_paths, all_test_data)]),\
                'Mismatched prediction and label file names'

            # Load predictions, sorted by file name
            sample_predictions = [np.clip(np.squeeze(np.load(p_f)), 0, 1) for p_f in pred_paths]

            all_results[p_dir.name] = {}
            for e_m in exp_config.eval_metrics:
                all_results[p_dir.name][e_m] = METRICS[e_m](sample_predictions, sample_labels, exp_config.eval_scales)

        print('Evaluation results:')
        print(json.dumps(all_results, indent=2))

        results_save_path = dset_pred_dir / 'results.json'
        print(f'Saving at ', results_save_path)

        with open(results_save_path, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate predictions on test sets of a given experiment")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser_args = parser.parse_args()

    main(make_exp_config(parser_args.EXP_PATH))
