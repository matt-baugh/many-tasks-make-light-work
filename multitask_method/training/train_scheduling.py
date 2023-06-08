import itertools
import time
from typing import List, Tuple, Optional

import logging
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from multitask_method.pos_encoding import PosEnc
from multitask_method.training.train_setup import construct_datasets
from multitask_method.training.training_dataset import TrainDatasetItem

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Training:

    def __init__(self, exp):

        self.exp = exp.__file__.split('/')[-1].rstrip('.py')

        # data
        self.curr_dset_coord = exp.curr_dset_coord
        self.other_dset_coord = exp.other_dset_coord
        self.labeller = exp.labeller
        self.fold = exp.args.fold
        self.num_train_tasks = exp.num_train_tasks
        self.pos_enc: Optional[PosEnc] = exp.pos_enc
        self.dset_size_cap = exp.dset_size_cap
        self.cache_pos_enc = exp.cache_pos_enc
        self.task_kwargs = exp.task_kwargs
        self.train_transforms = exp.train_transforms

        # learning & model
        self.model = exp.model
        self.optimizer = exp.optimizer
        self.lr_scheduler = exp.lr_scheduler
        self.lr = exp.lr
        self.weight_decay = exp.weight_decay
        self.criterion = exp.criterion
        self.accuracy = exp.accuracy
        self.start_epoch = 0
        self.batches_per_step = exp.batches_per_step
        self.epochs = exp.epochs
        self.batch_size = exp.batch_size
        self.shuffle = exp.shuffle
        self.num_workers = exp.num_workers

        # logging and checkpoint saving
        self.validate = exp.validate
        self.log_dir = exp.log_dir
        self.best_checkpoint = exp.best_checkpoint
        self.latest_checkpoint = exp.latest_checkpoint
        self.monitoring = exp.monitoring
        self.summary_writer = SummaryWriter(self.log_dir) if self.monitoring else None
        self.best_loss = exp.best_loss
        self.best_val_loss = exp.best_val_loss
        self.best_val_loss_global_step = 0
        self.best_accuracy = [np.inf] * len(self.accuracy)
        self.best_val_accuracy = [np.inf] * len(self.accuracy)
        self.num_log_images = 6
        self.early_stopping = exp.early_stopping
        self.early_stopping_patience = 5000
        self.ma_alpha = 0.93

        self.checkpoint_path = exp.args.checkpoint_path
        self.checkpoint = self._load_checkpoint()

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.start_epoch = self.checkpoint['epoch']

    def update_moving_average(self, average: float, curr: float) -> float:
        if np.isnan(average):
            return curr
        else:
            return self.ma_alpha * average + (1 - self.ma_alpha) * curr

    def _load_checkpoint(self):
        if self.checkpoint_path is None:
            return None
        logging.info(f"Loading model checkpoint from {self.checkpoint_path}")
        return torch.load(self.checkpoint_path)

    def train(self):

        # check cuda - GPU or CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model.to(device)
            torch.cuda.empty_cache()
            logging.info('Found cuda device!')
        else:
            device = torch.device("cpu")
            logging.info('Attention: No cuda device found - continuing on CPU!')

        # get data loader for training and validation
        logging.info('Loading dataset ...')
        train_dataset, val_dataset = construct_datasets(self.curr_dset_coord, self.other_dset_coord, self.fold,
                                                        None if self.cache_pos_enc else self.pos_enc,
                                                        self.num_train_tasks, self.task_kwargs, self.train_transforms,
                                                        self.dset_size_cap, self.labeller)

        train_loader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers)

        if self.cache_pos_enc and self.pos_enc is not None:
            sample_shape = train_loader.dataset[0][0].shape[1:]
            cached_pos_enc = torch.from_numpy(self.pos_enc(sample_shape)).float().to(device)[None]
            logging.info('Using cached positional encoding')
        else:
            cached_pos_enc = None

        # init optimiser
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = self.lr_scheduler(self.optimizer)

        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler_state_dict'])

        global_step = self.start_epoch * np.ceil(len(train_loader) / self.batches_per_step).astype(int)
        train_loss = np.nan
        train_accuracy = [np.nan] * len(self.accuracy)

        # training
        logging.info('Start Training')
        for epoch in range(self.start_epoch, self.epochs):

            start_time = time.time()
            self.model.train()

            train_examples = []

            step_losses = []
            step_metrics = []

            log_epoch = True  # epoch % 2 == 0
            for step, data in enumerate(train_loader, 0):  # type: int, TrainDatasetItem
                record_outputs = log_epoch and self.monitoring and \
                                 step % (len(train_loader) // self.num_log_images) == 0

                curr_losses, curr_metrics, tr_example = self.run_iteration(
                    cached_pos_enc, data, device, record_outputs, do_backprop=True, use_healthy=False)

                step_losses.extend(curr_losses)
                step_metrics.extend(curr_metrics)

                if (step + 1) % self.batches_per_step == 0 or step == len(train_loader):
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    avg_step_loss = np.average(step_losses)
                    avg_step_metrics = [sum(met) / len(met) for met in zip(*step_metrics)]

                    train_loss = self.update_moving_average(train_loss, avg_step_loss)
                    train_accuracy = [self.update_moving_average(tr_met, avg_step_met)
                                      for tr_met, avg_step_met in zip(train_accuracy, avg_step_metrics)]

                    del step_losses
                    del step_metrics

                    step_losses = []
                    step_metrics = []

                    if self.monitoring:
                        self.summary_writer.add_scalar("Train/loss_mse", avg_step_loss, global_step)
                        self.summary_writer.add_scalar("Train/loss_MA", train_loss, global_step)
                        self.summary_writer.add_scalar("Train/metric_bce", avg_step_metrics[0], global_step)
                        self.summary_writer.add_scalar("Train/metric_MA", train_accuracy[0], global_step)
                        self.summary_writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'],
                                                       global_step)

                    global_step += 1

                if record_outputs:
                    train_examples.append(tr_example)

                if step % 100 == 0 and train_loss is not None:
                    logging.info(
                        'Iteration: {}, train_loss_mse {:.4f}, metric_bce: {:.4f}'.format(step,
                                                                                          train_loss,
                                                                                          train_accuracy[0]))

            end_time = time.time()
            logging.info(
                "Epoch: {}, train_loss_mse: {:.4f}, bce: {:.4f}, time {:.2f}".format(epoch,
                                                                                     train_loss,
                                                                                     train_accuracy[0],
                                                                                     end_time - start_time))

            # saving checkpoints & writing summary
            if log_epoch:
                if self.validate:
                    logging.info("Validation at epoch {}".format(epoch))
                    start_time = time.time()
                    val_loss = 0
                    val_accuracy = [0] * len(self.accuracy)
                    val_examples = []
                    self.model.eval()
                    with torch.no_grad():
                        for step, data in tqdm(enumerate(val_loader, 0), 'Testing on validation set'):  # type: int,
                            # TrainDatasetItem
                            record_val_outputs = self.monitoring and step % (
                                    len(val_loader) // self.num_log_images) == 0

                            curr_val_losses, curr_val_metrics, v_example = self.run_iteration(
                                cached_pos_enc, data, device, record_val_outputs, do_backprop=False, use_healthy=True)

                            val_loss += sum(curr_val_losses)
                            val_accuracy = [sum(met_vals) for met_vals in zip(val_accuracy, *curr_val_metrics)]

                            if record_val_outputs:
                                val_examples.append(v_example)

                    end_time = time.time()

                    steps_in_val_ep = len(val_loader)
                    val_loss = val_loss / steps_in_val_ep
                    val_accuracy = [val_met / steps_in_val_ep for val_met in val_accuracy]
                    logging.info(
                        "Epoch: {}, val_loss: {:.4f}, mse: {:.4f}, time {:.2f}".format(epoch,
                                                                                       val_loss,
                                                                                       val_accuracy[0],
                                                                                       end_time - start_time))
                else:
                    val_loss = val_accuracy = val_examples = None

                if self.monitoring:
                    self.log_images(val_loss, val_accuracy, epoch, global_step, train_examples, val_examples)

                if self.best_checkpoint:
                    if bool(train_loss < self.best_loss):
                        self.best_loss = train_loss
                        self.save_checkpoint(epoch, train_loss, "best_model_loss.pt")
                    if self.validate and bool(val_loss < self.best_val_loss):
                        self.best_val_loss = val_loss
                        self.best_val_loss_global_step = global_step
                        if self.monitoring:
                            self.summary_writer.add_text('Best val loss log',
                                                         f'Step {global_step}: New best val loss: {val_loss:.4f}',
                                                         global_step)
                        self.save_checkpoint(epoch, val_loss, "best_model_val_loss.pt")
                    for a in range(len(train_accuracy)):
                        is_best = bool(train_accuracy[a] < self.best_accuracy[a])
                        self.best_accuracy[a] = min(train_accuracy[a], self.best_accuracy[a])
                        if is_best:
                            self.save_checkpoint(epoch, train_loss, "best_model_accuracy.pt")
                    if self.validate:
                        for a in range(len(val_accuracy)):
                            is_best = bool(val_accuracy[a] < self.best_val_accuracy[a])
                            self.best_val_accuracy[a] = min(val_accuracy[a], self.best_val_accuracy[a])
                            if is_best:
                                self.save_checkpoint(epoch, val_loss, "best_model_val_accuracy.pt")

                if self.latest_checkpoint:
                    self.save_checkpoint(epoch, train_loss, "latest_model_loss.pt")

                if self.validate and self.early_stopping and \
                        global_step > self.best_val_loss_global_step + self.early_stopping_patience:
                    print(f'Early stopping activated at epoch: {epoch} (global step: {global_step})')
                    print(f'Validation loss did not improve for {self.early_stopping_patience} epochs')
                    print(f'Best validation loss: {self.best_val_loss}')
                    print(f'Current validation loss: {val_loss}')
                    print('Stopping training.')
                    self.summary_writer.add_text('Best val loss log', 'Stop training as no improvement', global_step)
                    break

    def run_iteration(self, cached_pos_enc: torch.Tensor, data: TrainDatasetItem, device: torch.device,
                      record_outputs: bool, do_backprop: bool, use_healthy: bool) \
            -> (List[float], List[List[float]],
                Optional[Tuple[List[npt.NDArray], List[npt.NDArray], List[npt.NDArray]]]):

        normal_img, normal_pixel_label, _, aug_img, aug_pixel_label, _, img_pos_enc = data

        xs = [normal_img, aug_img] if use_healthy else [aug_img]
        ys = [normal_pixel_label, aug_pixel_label] if use_healthy else [aug_pixel_label]
        preds = []

        self.optimizer.zero_grad()

        curr_losses = []
        curr_metrics = []
        for curr_x, curr_y in zip(xs, ys):
            if self.cache_pos_enc:
                x_batch = torch.cat((curr_x.float().to(device),
                                     cached_pos_enc.expand((curr_x.shape[0], *cached_pos_enc.shape[1:]))),
                                    dim=1)
            else:
                x_batch = torch.cat((curr_x, img_pos_enc), dim=1).float().to(device)
            y_batch = curr_y.float().to(device)

            output = self.model(x_batch)

            del x_batch

            loss = self.criterion(output, y_batch)

            if do_backprop:
                loss.backward()

            curr_losses.append(loss.item())
            curr_metrics.append([metric_fn(output, y_batch).item() for metric_fn in self.accuracy])

            if record_outputs:
                preds.append(output[0].detach().cpu().numpy())

            del output

        example_tuple = ([x[0].numpy() for x in xs], [y[0].numpy() for y in ys], preds) if record_outputs else None

        return curr_losses, curr_metrics, example_tuple

    def predict(self, input_data: npt.NDArray[float]) -> npt.NDArray[float]:

        x_batch = torch.from_numpy(input_data).cuda()
        output = self.model(x_batch)

        return output

    def log_images(self, val_loss: float, val_accuracy: List[float], epoch: int, step: int,
                   train_examples: List[Tuple[List[npt.NDArray], List[npt.NDArray], List[npt.NDArray]]],
                   val_examples: List[Tuple[List[npt.NDArray], List[npt.NDArray], List[npt.NDArray]]]):

        if self.validate:
            self.summary_writer.add_scalar("Validation/loss", val_loss, step)
            self.summary_writer.add_scalar("Validation/mse", val_accuracy[0], step)

        imgs_to_log = [("Train", train_examples)]
        if self.validate:
            imgs_to_log.append(("Validation", val_examples))

        for data_name, examples in imgs_to_log:

            for example_num, (img, label, pred) in enumerate(examples):

                for img_name, image_to_show in [('input', img), ('label', label), ('pred', pred)]:
                    # Select first of healthy and unhealthy lists
                    image_to_show = np.concatenate(image_to_show, 0)
                    image_spatial_dims = image_to_show.shape[1:]

                    for i in range(len(image_to_show)):
                        if img_name == 'input':
                            image_to_show[i] = np.interp(image_to_show[i],
                                                         [image_to_show[i].min(), image_to_show[i].max()],
                                                         [0, 1])
                        else:
                            image_to_show[i] = np.clip(image_to_show[i], 0, 1)

                    crosssection_slices = [[slice(None) if d in dim_pair else image_spatial_dims[d] // 2
                                            for d in range(len(image_spatial_dims))]
                                           for dim_pair in itertools.combinations(range(len(image_spatial_dims)), 2)]

                    all_views = [image_to_show[(sam, *sl)]
                                 for sam, sl in itertools.product(list(range(len(image_to_show))), crosssection_slices)]
                    # Pad all images to the same size, keeping original image in the center
                    max_size = np.max(np.array([im.shape for im in all_views]), axis=0)[0]
                    all_views = [np.pad(im, tuple([(p // 2, p // 2) for p in max_size - np.array(im.shape)]))
                                 for im in all_views]

                    img_rows = [np.concatenate(all_views[i:i + 3], axis=1) for i in range(0, len(all_views), 3)]
                    img_grid = np.concatenate([np.pad(r, ((0, 0), (0, min(3, len(all_views)) * max_size - r.shape[1])))
                                               for r in img_rows],
                                              axis=0)

                    self.summary_writer.add_image(f'{data_name}_examples/{example_num}/{img_name}',
                                                  img_grid,
                                                  epoch,
                                                  dataformats='HW')

        # would be cool to add this: TensorBoardPlugin3D
        # plot_2d_or_3d_image(data=val_outputs, step=epoch, writer=writer, frame_dim=-1, tag="image")

    def save_checkpoint(self, epoch, loss, checkpoint_title):

        # saving checkpoint with given name
        model_path = self.log_dir / checkpoint_title
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
        }, model_path)
