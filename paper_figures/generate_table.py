from typing import List, Optional

import argparse
import json

import pandas as pd

from multitask_method.data.dataset_tools import DatasetCoordinator
from multitask_method.evaluation.eval_metrics import SLICE_WISE, SAMPLE_WISE, PIXEL_WISE, AP_SCORE
from multitask_method.paths import base_prediction_dir
from multitask_method.utils import make_exp_config

SLICE_WISE_CHOICE = 'mean'
SAMPLE_WISE_CHOICE = 'mean'


def format_metric(num):
    return f'{num * 100:.1f}'


def get_stats(arr):
    if len(arr) == 1:
        return f'{format_metric(arr[0])}'
    else:
        return f'{format_metric(arr.mean())}$\\pm${format_metric(arr.std())}'


def aggregate_index(ind):
    return ind[0], 'All' if ind[1].isnumeric() else 'Ensemble'


def generate_table(all_configs, use_cradl_results: bool, aggregate: bool, metric: Optional[List[str]]):
    all_config_results = []

    results_json = 'cradl_results.json' if use_cradl_results else 'results.json'

    for exp_config_path in all_configs:
        exp_config = make_exp_config(exp_config_path)
        exp_name = exp_config.name
        exp_display_name = int(exp_config.name.split('_')[-2])
        exp_display_name = f'{exp_display_name}/{5 - exp_display_name}'

        all_test_results = {}
        test_dset_name: str
        test_dset_coordinator: DatasetCoordinator

        for test_dset_name, test_dset_coordinator in exp_config.test_dsets.items():
            results_path = base_prediction_dir / exp_name / test_dset_name / results_json
            with open(results_path, 'r') as f:
                all_test_results[test_dset_name] = json.load(f)

        reorg_results = {}
        for test_dset_name, test_results in all_test_results.items():
            for fold_name, fold_results in test_results.items():
                for metric_name, metric_results in fold_results.items():
                    if metric is not None and metric_name not in metric:
                        continue
                    for scale_name, scale_results in metric_results.items():
                        if len(scale_results) == 1:
                            res_val = list(scale_results.values())[0]
                        elif scale_name == PIXEL_WISE:
                            assert metric_name == AP_SCORE

                            res_val = scale_results['In batches of 20']['ap_mean']
                        elif scale_name == SLICE_WISE:
                            res_val = scale_results[SLICE_WISE_CHOICE]
                        elif scale_name == SAMPLE_WISE:
                            res_val = scale_results[SAMPLE_WISE_CHOICE]
                        else:
                            raise NotImplementedError

                        multi_index_key = (scale_name, test_dset_name, metric_name)
                        if multi_index_key not in reorg_results:
                            reorg_results[multi_index_key] = {}

                        reorg_results[multi_index_key][(exp_display_name, fold_name)] = res_val

        results_df = pd.DataFrame(reorg_results, columns=pd.MultiIndex.from_tuples(reorg_results.keys()))
        results_df.index.rename(['Number of train tasks', 'Model'], inplace=True)
        results_df = results_df.sort_index(axis=1, level=[0, 1, 2], ascending=[False, True, True]).sort_index(axis=0)

        if aggregate:
            display_results = results_df.groupby(aggregate_index).aggregate(get_stats)
            display_results.index = pd.MultiIndex.from_tuples(display_results.index)
        else:
            display_results = results_df.applymap(format_metric)

        display_results = display_results.groupby(axis=1, level=[0, 1]).aggregate(lambda rs: '/'.join(rs))
        display_results = display_results.sort_index(axis=1, level=[0, 1], ascending=[False, True])
        display_results = display_results.sort_index(axis=0, ascending=True)

        all_config_results.append(display_results)

    all_results_table = pd.concat(all_config_results)
    # all_results_table.index.rename(['Task split', 'Fold'], inplace=True)
    return all_results_table
