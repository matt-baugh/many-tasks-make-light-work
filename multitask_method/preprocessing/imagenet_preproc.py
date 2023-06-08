from pathlib import Path

import numpy as np
from skimage import exposure, transform
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from tqdm import tqdm

from multitask_method.paths import base_data_input_dir

raw_root = Path('/vol/biodata/data/imagenet')
imagenet_output_dir = base_data_input_dir / 'imagenet'

full_res = 512
low_res = 256

imagenet_subset_size = 10000

norm_transforms = Compose([ToTensor(),
                           Grayscale(num_output_channels=1),
                           Resize((full_res, full_res), antialias=True)])


def make_torch_dset(root: Path, transforms=norm_transforms) -> ImageNet:
    return ImageNet(root, split='train', transform=transforms)


def imagenet_preproc_func(torch_img):
    np_img = torch_img.numpy().squeeze()
    np_img = exposure.equalize_hist(np_img, nbins=256)
    np_img = (np_img * 2) - 1
    return np_img


def preprocess_imagenet(root: Path):
    imagenet_output_dir.mkdir(exist_ok=True)

    full_res_out_dir = imagenet_output_dir / 'fullres'
    low_res_out_dir = imagenet_output_dir / 'lowres'

    full_res_out_dir.mkdir(exist_ok=True)
    low_res_out_dir.mkdir(exist_ok=True)

    dset = make_torch_dset(root)

    np.random.seed(123456789)
    indices = np.random.choice(len(dset), min(imagenet_subset_size, len(dset)), replace=False)

    for i in tqdm(indices, 'Preprocessing ImageNet'):
        torch_img, _ = dset[i]
        np_img = imagenet_preproc_func(torch_img)

        np.save(full_res_out_dir / f'{i}.npy', np_img)
        np.save(low_res_out_dir / f'{i}.npy', transform.resize(np_img, (low_res, low_res),
                                                               anti_aliasing=True, preserve_range=True))


if __name__ == '__main__':
    preprocess_imagenet(raw_root)
