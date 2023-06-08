import torch
import torch.optim as opt
import torch.optim.lr_scheduler as lr_sch
import torch.nn as nn
import numpy as np

from multitask_method.data.imagenet import ImagenetDatasetCoordinator
from multitask_method.data.vindr_cxr import VinDrCXRDatasetCoordinator
from multitask_method.evaluation.eval_metrics import SAMPLE_WISE, PIXEL_WISE, AP_SCORE, ROC_AUC_SCORE
from multitask_method.models.generic_UNet import Generic_UNet
from multitask_method.pos_encoding import CombinedEnc
from multitask_method.preprocessing.imagenet_preproc import imagenet_output_dir
from multitask_method.preprocessing.vindr_cxr_preproc import base_output_dir as vindr_root
from multitask_method.tasks.labelling import FlippedGaussianLabeller

# train data

curr_dset_coord = VinDrCXRDatasetCoordinator(vindr_root, False, ddad_split=True, train=True)
other_dset_coord = ImagenetDatasetCoordinator(imagenet_output_dir, False)
num_train_tasks = 1
dset_scale = 0.2
task_kwargs = {
    'intensity_task_scale': dset_scale,
    'min_push_dist': 0.5,
    'max_push_dist': 10
}
train_transforms = None

labeller = FlippedGaussianLabeller(dset_scale)
dset_size_cap = None

# test data
test_dsets = {
    'VINDR_TEST': VinDrCXRDatasetCoordinator(vindr_root, False, ddad_split=False, train=False),
    'DDAD_TEST': VinDrCXRDatasetCoordinator(vindr_root, False, ddad_split=True, train=False)
}

eval_scales = [SAMPLE_WISE, PIXEL_WISE]
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

NETWORK_DEPTH = 7
input_channels = 1 + (pos_enc.num_encoding_channels() if pos_enc is not None else 0)
model = Generic_UNet(input_channels=input_channels, base_num_features=Generic_UNet.BASE_NUM_FEATURES,
                     num_pool=NETWORK_DEPTH - 1, num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=conv_op,
                     final_nonlin=torch.nn.Identity(),
                     norm_op=norm_op, norm_op_kwargs={'eps': 1e-5, 'affine': True},
                     dropout_op=dropout_op, dropout_op_kwargs={'p': 0, 'inplace': True},
                     nonlin=nn.GELU, nonlin_kwargs={},
                     deep_supervision=False, convolutional_pooling=True, convolutional_upsampling=False)
optimizer = opt.AdamW
lr = 5e-3
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
batch_size = 50
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
