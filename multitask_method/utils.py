import logging
from pathlib import Path
import shutil

import numpy.typing as npt
import SimpleITK as sitk
from importlib.machinery import SourceFileLoader

BRAIN_SCALE = 0.5
ABDOM_SCALE = 0.25


def make_log_folder(log_dir: Path, overwrite_folder: bool):
    try:
        log_dir.mkdir(parents=True)
        logging.info('Made new log dir!')
    except FileExistsError:
        logging.info('Log dir already existent! Rename experiment or remove dir! Overwrite with "y".')
        if overwrite_folder or input("Continue? [y/n]") == "y":
            shutil.rmtree(log_dir)
            log_dir.mkdir()
            return
        else:
            raise


def make_exp_config(exp_file):
    # get path to experiment
    exp_name = exp_file.split('/')[-1].rstrip('.py')

    # import experiment configuration
    exp_config = SourceFileLoader(exp_name, exp_file).load_module()
    exp_config.name = exp_name
    return exp_config


def get_resample_factor(dataset) -> float:
    return BRAIN_SCALE if dataset == 'brain' else ABDOM_SCALE


def resample_sitk(dataset, img: sitk.Image) -> sitk.Image:
    scale = BRAIN_SCALE if dataset == 'brain' else ABDOM_SCALE

    return sitk.Resample(img,
                         size=[int(s * scale) for s in img.GetSize()],
                         outputDirection=img.GetDirection(),
                         outputSpacing=[sp / scale for sp in img.GetSpacing()],
                         outputOrigin=img.GetOrigin())


def resample_array(dataset, img: npt.NDArray) -> npt.NDArray:
    return sitk.GetArrayFromImage(resample_sitk(dataset, sitk.GetImageFromArray(img)))


def load_nii_gz(folder: Path, file: str) -> npt.NDArray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(folder / file)))
