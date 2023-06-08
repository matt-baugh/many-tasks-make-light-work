import argparse
import logging
import shutil


from multitask_method.paths import base_log_dir, exp_log_dir
from multitask_method.utils import make_exp_config, make_log_folder
from multitask_method.training.train_scheduling import Training


def main(exp):
    logging.info('--------------------------------------------------------------')
    logging.info('     Running Experiment: %s', exp.name)
    logging.info('--------------------------------------------------------------')

    logging.info('     Model:')
    logging.info(exp.model)

    proc = Training(exp)
    logging.info('--------------------------------------------------------------')
    logging.info('     Training ')
    logging.info('--------------------------------------------------------------')
    proc.train()


if __name__ == '__main__':
    # parser for arguments
    parser = argparse.ArgumentParser(description="TRAINING")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("fold", type=int, help="Choose fold to train on.")
    parser.add_argument("--checkpoint_path", type=str, help="Optional path to checkpoint")
    parser.add_argument("--overwrite_folder", help="Automatically overwrite log folder", action="store_true")
    args = parser.parse_args()

    exp_config = make_exp_config(args.EXP_PATH)
    exp_config.args = args

    # make root log dir
    base_log_dir.mkdir(parents=True, exist_ok=True)

    exp_config.log_dir = exp_log_dir(exp_config.name, args.fold)
    make_log_folder(exp_config.log_dir, args.overwrite_folder)

    # copy experiment to log dir
    shutil.copy(exp_config.__file__, exp_config.log_dir)

    main(exp=exp_config)
