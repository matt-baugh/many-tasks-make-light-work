from pathlib import Path
from tqdm import tqdm

import SimpleITK as sitk

from multitask_method.utils import resample_sitk

dataset_is_brain = False

dataset_folder = 'Brain/brain' if dataset_is_brain else 'Abdomen/abdom'
max_bg = 0 if dataset_is_brain else 0.03

input_folder = Path(f'/vol/biomedic2/MOOD/{dataset_folder}_train/')
output_folder = Path(f'/vol/biomedic2/MOOD/{dataset_folder}_train_lowres/')
mask_folder = Path(f'/vol/biomedic2/MOOD/{dataset_folder}_mask_lowres/')

assert input_folder.is_dir()
output_folder.mkdir(exist_ok=True)
mask_folder.mkdir(exist_ok=True)

print('Input folder: ', input_folder)
print('Output folder: ', output_folder)
print('Output mask folder: ', mask_folder)

for f in tqdm(input_folder.iterdir()):
    sitk_img = sitk.ReadImage(str(f))

    resampled_img = resample_sitk('brain' if dataset_is_brain else 'abdom', sitk_img)
    resampled_img_mask = 1 - sitk.ConnectedThreshold(resampled_img, [(0, 0, 0)], lower=0, upper=max_bg)

    sitk.WriteImage(resampled_img, str(output_folder / f.name))
    sitk.WriteImage(resampled_img_mask, str(mask_folder / f.name))
