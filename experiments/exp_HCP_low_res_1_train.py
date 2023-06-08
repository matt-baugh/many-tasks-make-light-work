from pathlib import Path

import torch
import torch.optim as opt
import torch.optim.lr_scheduler as lr_sch
import torch.nn as nn
import numpy as np

from multitask_method.data.mood import MOODDatasetCoordinator
from multitask_method.data.preprocessed_mri import PreprocessedMRIDatasetCoordinator
from multitask_method.evaluation.eval_metrics import SLICE_WISE, PIXEL_WISE, AP_SCORE, ROC_AUC_SCORE
from multitask_method.models.generic_UNet import Generic_UNet
from multitask_method.paths import base_data_input_dir
from multitask_method.pos_encoding import CombinedEnc
from multitask_method.tasks.labelling import FlippedGaussianLabeller

# train data
data_size = np.array([80, 112, 80])

hcp_root = base_data_input_dir / 'hcp'
# Data modality 1 is T2w
curr_dset_coord = PreprocessedMRIDatasetCoordinator(hcp_root, [1], False, data_size)
# IMPERIAL: '/vol/biomedic2/MOOD/' FAU: '/vol/ideadata/ed52egek/data/mooddata/'
mood_root = Path('/vol/biomedic2/MOOD/')
other_dset_coord = MOODDatasetCoordinator(mood_root, 'Abdomen', False, z_normalise=True)
num_train_tasks = 1
dset_scale = 0.2
task_kwargs = {
    'intensity_task_scale': dset_scale,
    'min_push_dist': 0.5,
    'max_push_dist': 5
}


def train_transforms(img, mask):
    if np.random.rand() < 0.5:
        return img, mask
    else:
        return img[..., ::-1], mask[..., ::-1]

labeller = FlippedGaussianLabeller(dset_scale)
dset_size_cap = None

# test data
test_dsets = {
    'ISLES': PreprocessedMRIDatasetCoordinator(base_data_input_dir / 'isles2015', [1], False, data_size,
                                               foreground_mask=False),
    'BRATS': PreprocessedMRIDatasetCoordinator(base_data_input_dir / 'brats17', [1], False, data_size,
                                               foreground_mask=False),
}
eval_scales = [SLICE_WISE, PIXEL_WISE]
eval_metrics = [AP_SCORE, ROC_AUC_SCORE]

# learning & model

data_dims = curr_dset_coord.dataset_dimensions()
pos_enc = CombinedEnc(data_dims, 32, 12)
cache_pos_enc = True
validate = True

if data_dims == 3:
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
elif data_dims == 2:
    conv_op = nn.Conv2d
    dropout_op = nn.Dropout2d
    norm_op = nn.InstanceNorm2d
else:
    raise Exception('INVALID DATA DIMENSIONALITY')

NETWORK_DEPTH = 5
input_channels = 1 + (pos_enc.num_encoding_channels() if pos_enc is not None else 0)
model = Generic_UNet(input_channels=input_channels, base_num_features=Generic_UNet.BASE_NUM_FEATURES,
                     num_pool=NETWORK_DEPTH - 1, num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=conv_op,
                     final_nonlin=torch.nn.Identity(),
                     norm_op=norm_op, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                     dropout_op=dropout_op, dropout_op_kwargs={'p': 0, 'inplace': True},
                     nonlin=nn.GELU, nonlin_kwargs={},
                     deep_supervision=False, convolutional_pooling=True, convolutional_upsampling=False)
optimizer = opt.AdamW
lr = 1e-3
min_lr_factor = 1 / 20
weight_decay = 0

switch_schedule = 100
schedule_factor = 0.9997122182804811  # Period 8000
constant_schedule = np.round(np.log(min_lr_factor) / np.log(schedule_factor)) + switch_schedule


def lr_scheduler(my_opt):
    return lr_sch.SequentialLR(my_opt,
                               [lr_sch.LinearLR(my_opt, start_factor=1e-2, total_iters=switch_schedule),
                                lr_sch.ExponentialLR(my_opt, schedule_factor),
                                lr_sch.ConstantLR(my_opt, min_lr_factor, total_iters=torch.inf)],
                               milestones=[switch_schedule, constant_schedule])


criterion = nn.functional.mse_loss
accuracy = [lambda output, y: torch.nn.functional.binary_cross_entropy(torch.clip(output, 0, 1), y)]
epochs = 2000
batch_size = 8
batches_per_step = 1
shuffle = True
num_workers = 0

# logging and checkpoint saving
best_checkpoint = True
latest_checkpoint = True
monitoring = True
best_loss = np.inf
best_val_loss = np.inf
early_stopping = True
