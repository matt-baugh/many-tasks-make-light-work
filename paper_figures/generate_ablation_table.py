import os

import pandas as pd

from paper_figures.generate_table import generate_table

from experiments import exp_HCP_low_res_1_train, exp_HCP_low_res_2_train, exp_HCP_low_res_3_train, exp_HCP_low_res_4_train,\
    exp_VINDR_low_res_1_train, exp_VINDR_low_res_2_train, exp_VINDR_low_res_3_train,\
    exp_VINDR_low_res_4_train

from multitask_method.evaluation.eval_metrics import AP_SCORE

all_experiments = {
    'Brain': [
        exp_HCP_low_res_1_train,
        exp_HCP_low_res_2_train,
        exp_HCP_low_res_3_train,
        exp_HCP_low_res_4_train
    ],
    'VinDR-CXR': [
        exp_VINDR_low_res_1_train,
        exp_VINDR_low_res_2_train,
        exp_VINDR_low_res_3_train,
        exp_VINDR_low_res_4_train
    ]
}

if __name__ == '__main__':

    tables = [(table_name, generate_table([os.path.abspath(e_c.__file__) for e_c in exp_configs], False, True,
                                          [AP_SCORE]))
              for table_name, exp_configs in all_experiments.items()]

    print('Ablation table')
    abl_table = pd.concat([table for _, table in tables], axis=1, keys=[table_name for table_name, _ in tables])

    print(abl_table.to_latex(multirow=True, escape=False))
