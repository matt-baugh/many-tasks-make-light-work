from pathlib import Path

base_data_input_dir = Path('/mounted-data/my-biomedic3/multitask_method_data')
base_prediction_dir = Path('/mounted-data/my-biomedic3/multitask_method_preds_tmp')
base_log_dir = Path('/mounted-data/my-biomedic3/multitask-method/log')

ensemble_dir = 'ensemble'


def exp_log_dir(exp_name, fold):
    return base_log_dir / exp_name / str(fold)
