from typing import List, Literal

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

SAMPLE_WISE = 'sample_wise'
SLICE_WISE = 'slice_wise'
PIXEL_WISE = 'pixel_wise'

ALL_EVAL_SCALES = [SAMPLE_WISE, SLICE_WISE, PIXEL_WISE]

AP_SCORE = 'average_precision_score'
ROC_AUC_SCORE = 'roc_auc_score'

AP_BATCH_SIZES = [20, None]
AP_BATCH_REPEATS = 10

PRED_AGGREGATION_METHODS = {
    'mean': np.mean,
    '75_percentile': lambda p: np.percentile(p, 75),
    'mean_top_25_perc': lambda p: np.mean(p[np.argpartition(p, -int(len(p) * 0.25))[-int(len(p) * 0.25):]]),
}


def get_zimmerer_slice_labels(all_sample_labels: List[npt.NDArray[int]]) -> npt.NDArray[int]:
    all_slices_labels = np.concatenate(all_sample_labels, axis=0)

    # Use same proportional threshold as Zimmerer et al. Proportional as slices differ in size.
    slice_level_labels = np.mean(all_slices_labels, axis=tuple(range(len(all_slices_labels.shape)))[1:]) > (5 / (128 ** 2))
    slice_level_labels = slice_level_labels.astype(int)

    return slice_level_labels


def flatten_all(arrays: List[npt.NDArray]) -> npt.NDArray:
    return np.concatenate([np.ravel(a) for a in arrays])


CONVERT_LABELS = {
    SAMPLE_WISE: lambda sample_labels: np.array([np.max(s_l) for s_l in sample_labels]),
    SLICE_WISE: get_zimmerer_slice_labels,
    PIXEL_WISE: flatten_all
}


def eval_ap_score(all_sample_predictions: List[npt.NDArray[float]], all_sample_labels: List[npt.NDArray[int]],
                  eval_levels: List[str]) -> dict:
    assert len(all_sample_predictions) == len(all_sample_labels), 'Pred-label mismatch'

    res = {}

    for lvl in eval_levels:
        res[lvl] = {}
        level_labels = CONVERT_LABELS[lvl](all_sample_labels)

        if lvl == SAMPLE_WISE:

            assert all([s_l in [0, 1] for s_l in level_labels])

            for agg_name, agg_fn in PRED_AGGREGATION_METHODS.items():
                res[lvl][agg_name] = average_precision_score(level_labels,
                                                             [agg_fn(s_p) for s_p in all_sample_predictions])
        elif lvl == SLICE_WISE:

            all_slice_predictions = np.concatenate(all_sample_predictions, axis=0)
            for agg_name, agg_fn in PRED_AGGREGATION_METHODS.items():
                res[lvl][agg_name] = average_precision_score(level_labels,
                                                             [agg_fn(s_p) for s_p in all_slice_predictions])
        elif lvl == PIXEL_WISE:

            for ap_bs in AP_BATCH_SIZES:
                if ap_bs is None:
                    res[lvl]['Over entire dset'] = average_precision_score(level_labels,
                                                                           flatten_all(all_sample_predictions))
                    continue

                # Otherwise, do compute ap_score AP_BATCH_REPEATS times using different dataset orderings
                ap_repeats = []
                for i in range(AP_BATCH_REPEATS):

                    indices = np.random.default_rng(seed=i).permutation(len(all_sample_labels))

                    shuffled_sample_labels = [all_sample_labels[i].flatten() for i in indices]
                    shuffled_sample_predictions = [all_sample_predictions[i].flatten() for i in indices]

                    curr_aps = []
                    for chunk_start in range(0, len(shuffled_sample_labels), ap_bs):
                        chunk_labels = np.concatenate(shuffled_sample_labels[chunk_start: chunk_start + ap_bs])
                        chunk_preds = np.concatenate(shuffled_sample_predictions[chunk_start: chunk_start + ap_bs])
                        curr_aps.append(average_precision_score(chunk_labels, chunk_preds))
                    ap_repeats.append(np.mean(curr_aps))

                ap_bs_name = f'In batches of {ap_bs}'
                res[lvl][ap_bs_name] = {
                    'ap_repeats': ap_repeats,
                    'ap_mean': np.mean(ap_repeats),
                    'ap_std': np.std(ap_repeats)
                }

    return res


def eval_roc_auc_score(all_sample_predictions: List[npt.NDArray[float]], all_sample_labels: List[npt.NDArray[int]],
                       eval_levels: List[str]) -> dict:
    assert len(all_sample_predictions) == len(all_sample_labels), 'Pred-label mismatch'

    res = {}

    for lvl in eval_levels:
        res[lvl] = {}
        level_labels = CONVERT_LABELS[lvl](all_sample_labels)

        if lvl == SAMPLE_WISE:
            assert all([s_l in [0, 1] for s_l in level_labels])

            for agg_name, agg_fn in PRED_AGGREGATION_METHODS.items():
                res[lvl][agg_name] = roc_auc_score(level_labels,
                                                   [agg_fn(s_p) for s_p in all_sample_predictions])
        elif lvl == SLICE_WISE:

            all_slice_predictions = np.concatenate(all_sample_predictions, axis=0)
            for agg_name, agg_fn in PRED_AGGREGATION_METHODS.items():
                res[lvl][agg_name] = roc_auc_score(level_labels,
                                                   [agg_fn(s_p) for s_p in all_slice_predictions])
        elif lvl == PIXEL_WISE:
            res[lvl]['Over entire dset'] = roc_auc_score(level_labels,
                                                         flatten_all(all_sample_predictions))

    return res


METRICS = {
    AP_SCORE: eval_ap_score,
    ROC_AUC_SCORE: eval_roc_auc_score
}

METRICS_RANDOM_BASELINE = {
    AP_SCORE: lambda label_arr: np.mean(label_arr),
    ROC_AUC_SCORE: lambda label_arr: 0.5
}


def get_test_anom_data(test_dset_coordinator):
    test_dset_container = test_dset_coordinator.make_container(list(range(len(test_dset_coordinator))))
    all_test_data = sorted([test_dset_container[i] for i in range(len(test_dset_container))], key=lambda t: t[2])
    sample_labels = [np.squeeze((s_anom_mask != 0).astype(int)) for _, s_anom_mask, _ in all_test_data]
    return all_test_data, sample_labels
