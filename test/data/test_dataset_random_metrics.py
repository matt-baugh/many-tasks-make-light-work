import pandas as pd

from multitask_method.evaluation.eval_metrics import get_test_anom_data, CONVERT_LABELS, METRICS_RANDOM_BASELINE
from test.example_configs import example_hcp_config, example_vindr_config


def get_random_baselines(exp_config):
    eval_scales = exp_config.eval_scales
    eval_metrics = exp_config.eval_metrics

    all_test_labels = [(test_dset_name, get_test_anom_data(test_dset_coord)[1])
                       for test_dset_name, test_dset_coord in exp_config.test_dsets.items()]

    columns = []
    results = []

    for scale in eval_scales:

        for test_dset_name, test_dset_labels in all_test_labels:

            scale_labels = CONVERT_LABELS[scale](test_dset_labels)

            curr_results = []
            for met in eval_metrics:
                curr_results.append(f'{METRICS_RANDOM_BASELINE[met](scale_labels):.3f}')

            results.append('/'.join(curr_results))
            columns.append((scale, test_dset_name))

    results_df = pd.DataFrame([results], index=['Random'], columns=pd.MultiIndex.from_tuples(columns))
    display_results = results_df.sort_index(1, level=[0, 1], ascending=[False, True])
    print(display_results.to_latex())


def test_get_brain_baselines():
    get_random_baselines(example_hcp_config)


def test_get_vindr_baselines():
    get_random_baselines(example_vindr_config)

