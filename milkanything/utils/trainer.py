import builtins

_original_open = builtins.open

def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if 'b' not in mode and encoding is None:
        encoding = 'utf-8'
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

builtins.open = _utf8_open

import os
import time
import torch
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from milkanything.nets.discriminator import Discriminator
from milkanything.nets.generator import Generator
from milkanything.utils.dataset import CycleGANDataset

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    print("Warning: 'pillow-heif' is not installed.")


class MilkAnythingTrainer:
    def __init__(self, G_A2B: Generator, G_B2A: Generator, D_A: Discriminator, D_B: Discriminator,
                 train_dataset: CycleGANDataset, val_dataset: CycleGANDataset, batch_size: int = 4,
                 num_epochs: int = 200, learning_rate: float = 2e-4, lambda_cyc: float = 10.0,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 num_workers: int = 4, save_dir: str = "./output",
                 auto_save: bool = True, use_amp: bool = True, dtype: torch.dtype = torch.float16):

        self.G_A2B = G_A2B
        self.G_B2A = G_B2A
        self.D_A = D_A
        self.D_B = D_B

        self.auto_save = auto_save
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.lambda_cyc = lambda_cyc

        if use_amp and device.type == 'cuda':
            torch.cuda.enable_graphs = True
            torch._C._accelerator_setAllocatorSettings('max_split_size_mb:128')
            high_priority_stream = torch.cuda.Stream(priority=-1)
            torch.cuda.set_stream(high_priority_stream)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        self.use_amp = use_amp and torch.cuda.is_available() and device.type == 'cuda'
        self.dtype = dtype

        self.scaler_G = torch.amp.GradScaler(enabled=self.use_amp)
        self.scaler_D_A = torch.amp.GradScaler(enabled=self.use_amp)
        self.scaler_D_B = torch.amp.GradScaler(enabled=self.use_amp)

        self.G_A2B.to(device, non_blocking=True)
        self.G_B2A.to(device, non_blocking=True)
        self.D_A.to(device, non_blocking=True)
        self.D_B.to(device, non_blocking=True)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True, prefetch_factor=4, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True, prefetch_factor=4, drop_last=True
        )

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.learning_rate, betas=(0.5, 0.999),
                                              weight_decay=1e-4)
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.learning_rate, betas=(0.5, 0.999),
                                              weight_decay=1e-4)

        self.scheduler_G = CosineAnnealingWarmRestarts(self.optimizer_G, T_0=10, T_mult=2, eta_min=1e-6)

        self.g_loss_history = []
        self.d_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.save_dir = save_dir
        os.makedirs(os.path.join(self.save_dir, 'checkpoints'), exist_ok=True)

    def train_epoch(self, epoch):
        self.G_A2B.train()
        self.G_B2A.train()
        self.D_A.train()
        self.D_B.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        num_batches = len(self.train_loader)

        train_pbar = tqdm.tqdm(enumerate(self.train_loader), total=num_batches,
                               desc=f'Training Epoch {epoch + 1}/{self.num_epochs}',
                               unit='batch', leave=False)

        for batch_idx, batch in train_pbar:
            real_A = batch["A"].to(self.device, non_blocking=True)
            real_B = batch["B"].to(self.device, non_blocking=True)

            self.optimizer_G.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp,
                                    cache_enabled=True):
                fake_B = self.G_A2B(real_A)
                pred_fake_B = self.D_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

                fake_A = self.G_B2A(real_B)
                pred_fake_A = self.D_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

                rec_A = self.G_B2A(fake_B)
                loss_cycle_A = self.criterion_cycle(rec_A, real_A)

                rec_B = self.G_A2B(fake_A)
                loss_cycle_B = self.criterion_cycle(rec_B, real_B)

                loss_G = loss_GAN_A2B + loss_GAN_B2A + self.lambda_cyc * (loss_cycle_A + loss_cycle_B)

            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()

            self.optimizer_D_A.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp,
                                    cache_enabled=True):
                pred_real_A = self.D_A(real_A)
                loss_D_real_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

                pred_fake_A = self.D_A(fake_A.detach())
                loss_D_fake_A = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

            self.scaler_D_A.scale(loss_D_A).backward()
            self.scaler_D_A.step(self.optimizer_D_A)
            self.scaler_D_A.update()

            self.optimizer_D_B.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp,
                                    cache_enabled=True):
                pred_real_B = self.D_B(real_B)
                loss_D_real_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

                pred_fake_B = self.D_B(fake_B.detach())
                loss_D_fake_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            self.scaler_D_B.scale(loss_D_B).backward()
            self.scaler_D_B.step(self.optimizer_D_B)
            self.scaler_D_B.update()

            running_g_loss += loss_G.item()
            running_d_loss += (loss_D_A.item() + loss_D_B.item())
            train_pbar.set_postfix({
                "G_Loss": f"{loss_G.item():.4f}",
                "D_Loss": f"{(loss_D_A + loss_D_B).item():.4f}"
            })

        return running_g_loss / num_batches, running_d_loss / num_batches

    def validate(self):
        self.G_A2B.eval()
        self.G_B2A.eval()
        running_cycle_loss = 0.0
        num_batches = len(self.val_loader)

        val_pbar = tqdm.tqdm(enumerate(self.val_loader), total=num_batches, desc='Validating', unit='batch',
                             leave=False)

        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                real_A = batch["A"].to(self.device, non_blocking=True)
                real_B = batch["B"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp,
                                        cache_enabled=True):
                    fake_B = self.G_A2B(real_A)
                    rec_A = self.G_B2A(fake_B)
                    loss_cycle = self.criterion_cycle(rec_A, real_A)

                running_cycle_loss += loss_cycle.item()
                val_pbar.set_postfix({'cycle_loss': loss_cycle.item()})

        return running_cycle_loss / num_batches

    def train(self):
        print(f"CycleGAN with Custom Backbone Discriminator training on: {self.device}")
        epoch_pbar = tqdm.tqdm(range(self.num_epochs), desc='Training process', unit='epoch', position=0)

        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            self.g_loss_history.append(avg_g_loss)
            self.d_loss_history.append(avg_d_loss)

            avg_val_loss = self.validate()
            self.val_loss_history.append(avg_val_loss)

            self.scheduler_G.step()

            epoch_time = time.time() - epoch_start_time

            epoch_pbar.set_postfix({
                'G_loss': f'{avg_g_loss:.4f}',
                'D_loss': f'{avg_d_loss:.4f}',
                'Val_cycle_loss': f'{avg_val_loss:.4f}',
                'time': f'{epoch_time:.1f}s'
            })

            if avg_val_loss < self.best_val_loss and self.auto_save:
                self.best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.save_dir, 'checkpoints/best_cyclegan.pth')
                torch.save({
                    'epoch': epoch,
                    'G_A2B_state_dict': self.G_A2B.state_dict(),
                    'G_B2A_state_dict': self.G_B2A.state_dict(),
                    'D_A_state_dict': self.D_A.state_dict(),
                    'D_B_state_dict': self.D_B.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, best_model_path)

        epoch_pbar.close()
        self._plot_loss_curves()
        print(f"Training Complete. Best validation reconstruction loss: {self.best_val_loss:.6f}")
        return self.g_loss_history, self.d_loss_history, self.val_loss_history

    def _plot_loss_curves(self, dpi=150, figure_size=(10, 6)):
        epochs = list(range(1, len(self.g_loss_history) + 1))
        plt.figure(figsize=figure_size, dpi=dpi)
        plt.plot(epochs, self.g_loss_history, 'b-', label='Generator Loss')
        plt.plot(epochs, self.d_loss_history, 'r-', label='Discriminator Loss')
        plt.plot(epochs, self.val_loss_history, 'g-', label='Val Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CycleGAN Training Curves')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'cyclegan_curves.png'), bbox_inches='tight', dpi=dpi)
        plt.close()