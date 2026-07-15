import copy
import os
import statistics
import time

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from nets.discriminator import Discriminator
from utils.dataset import DiscriminatorDataset


class DiscriminatorTrainer:
    def __init__(self, model: Discriminator, train_dataset: DiscriminatorDataset, eval_dataset: DiscriminatorDataset, batch_size: int = 16,
                 num_epochs: int = 200, learning_rate: float = 1e-3,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 patience: int = 50, num_workers: int = 8,
                 save_dir: str = "./output", auto_save: bool = False, enable_early_stop: bool = False,
                 use_amp: bool = True, dtype: torch.dtype = torch.float16):
        self.model = model
        self.auto_save = auto_save
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.enable_early_stop = enable_early_stop

        if use_amp:
            torch.cuda.enable_graphs = True
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch._C._accelerator_setAllocatorSettings('max_split_size_mb:128')
            torch.cuda.memory.caching_allocator_alloc(0)
            high_priority_stream = torch.cuda.Stream(priority=-1)
            torch.cuda.set_stream(high_priority_stream)

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        self.use_amp = use_amp and torch.cuda.is_available() and device.type == 'cuda'
        self.dtype = dtype
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.device = device
        self.model.to(device, non_blocking=True)

        if device.type == 'cuda':
            self.model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False
            )

        self.train_dataset = train_dataset
        self.val_dataset = eval_dataset

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers = True,
            prefetch_factor = 8,
            drop_last = True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers = True,
            prefetch_factor = 8,
            drop_last = True,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.patience = patience

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        train_pbar = tqdm.tqdm(enumerate(self.train_loader), total=num_batches,
                               desc=f'Training Epoch {epoch + 1}/{self.num_epochs}',
                               unit='batch', leave=False)

        for batch_idx, (images, labels) in train_pbar:
            images = images.to(self.device, non_blocking=True)
            labels_xy = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp, cache_enabled=True):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_xy)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            train_pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = running_loss / num_batches
        return avg_train_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        num_batches = len(self.val_loader)

        val_pbar = tqdm.tqdm(enumerate(self.val_loader), total=num_batches, desc='Validating', unit='batch',
                             leave=False)

        with torch.no_grad():
            for batch_idx, (images, labels_xy) in val_pbar:
                images = images.to(self.device, non_blocking=True)
                labels_xy = labels_xy.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp, cache_enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels_xy)

                running_loss += loss.item()

                val_pbar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = running_loss / num_batches
        return avg_val_loss

    def train(self, resume_epoch=0):
        print(f"Training on device: {self.device}")
        print(f"Using AMP: {self.use_amp}")
        if self.use_amp:
            print(f"AMP dtype: {self.dtype}")
        print(f"Size of train dataset: {len(self.train_dataset)}, Size of validation dataset: {len(self.val_dataset)}")
        print(f"Number of epochs: {self.num_epochs}")
        print("-" * 60)

        config = {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'save_dir': self.save_dir,
            'patience': self.patience,
            'use_amp': self.use_amp,
            'dtype': str(self.dtype)
        }

        start_epoch = resume_epoch

        epoch_pbar = tqdm.tqdm(range(start_epoch, self.num_epochs), desc='Training process', unit='epoch', position=0)

        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            avg_train_loss = self.train_epoch(epoch)
            self.train_loss_history.append(avg_train_loss)

            avg_val_loss = self.validate()
            self.val_loss_history.append(avg_val_loss)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'time': f'{epoch_time:.2f}s',
                'amp': str(self.use_amp)
            })

            if avg_val_loss < self.best_val_loss and self.auto_save == True:
                self.best_val_loss = avg_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0

                best_model_path = os.path.join(self.save_dir,
                                               f'best_model_epoch{epoch + 1}_val_loss{avg_val_loss:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                    'config': config
                }, best_model_path)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience and self.enable_early_stop == True:
                print(f'\nEarly Stop Warning, triggered in epoch {epoch + 1}')
                epoch_pbar.close()
                break

            if (epoch + 1) % 5 == 0 and self.auto_save == True:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_loss_history': self.train_loss_history,
                    'val_loss_history': self.val_loss_history,
                    'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                    'config': config
                }, checkpoint_path)

        if not epoch_pbar.disable:
            epoch_pbar.close()

        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': config,
            'best_val_loss': self.best_val_loss
        }, final_model_path)
        print(f'\nFinal model save to: {final_model_path}')

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print('Train completed, loading best model')

        print(f'Best validation loss: {self.best_val_loss:.6f}')
        if self.train_loss_history:
            print(f'Average training loss: {statistics.mean(self.train_loss_history):.6f}')
        if self.val_loss_history:
            print(f'Average validation loss: {statistics.mean(self.val_loss_history):.6f}')

        self._plot_smooth_loss_curves()

        return self.train_loss_history, self.val_loss_history

    def load_checkpoint(self, checkpoint_path, model, map_location=None):
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        config = checkpoint.get('config', {})

        print(f"Loading model from checkpoint: {checkpoint_path}")
        print(f"Epochs: {epoch}, Config: {config}")

        return epoch

    def _plot_smooth_loss_curves(self, window_size=5, show_plot=True, dpi=150, figure_size=(12, 8)):
        epochs = list(range(1, len(self.train_loss_history) + 1))

        def moving_average(data, size):
            if len(data) < size:
                return data
            return np.convolve(data, np.ones(size) / size, mode='valid')

        train_smooth = moving_average(self.train_loss_history, window_size)
        val_smooth = moving_average(self.val_loss_history, window_size)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, dpi=dpi, gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(epochs, self.train_loss_history, 'b-', linewidth=1.5, label='Train Loss (Raw)', alpha=0.5)
        ax1.plot(epochs, self.val_loss_history, 'r-', linewidth=1.5, label='Val Loss (Raw)', alpha=0.5)

        smooth_epochs = list(range(window_size, len(self.train_loss_history) + 1))
        ax1.plot(smooth_epochs, train_smooth, 'b-', linewidth=2.5, label=f'Train Loss (Smooth, window={window_size})')
        ax1.plot(smooth_epochs, val_smooth, 'r-', linewidth=2.5, label=f'Val Loss (Smooth, window={window_size})')

        if self.best_val_loss < float('inf'):
            best_epoch = self.val_loss_history.index(min(self.val_loss_history)) + 1
            best_val_loss = min(self.val_loss_history)
            ax1.scatter(best_epoch, best_val_loss, color='gold', s=200,
                        marker='*', edgecolors='black', linewidth=2,
                        zorder=5, label=f'Best Val Loss: {best_val_loss:.6f}')

            ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss Curves (Raw vs Smooth)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right')

        if len(self.train_loss_history) == len(self.val_loss_history):
            loss_ratio = [t / v if v > 0 else 0 for t, v in zip(self.train_loss_history, self.val_loss_history)]
            ax2.plot(epochs, loss_ratio, 'g-', linewidth=2, label='Train Loss / Val Loss', alpha=0.8)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Ratio = 1')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss Ratio', fontsize=12)
            ax2.set_title('Train Loss / Val Loss Ratio', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=10, loc='upper right')

        plt.tight_layout()

        smooth_save_path = os.path.join(self.save_dir, 'loss_curves.png')
        plt.savefig(smooth_save_path, bbox_inches='tight', dpi=dpi)

        if show_plot:
            plt.show()
        else:
            plt.close()