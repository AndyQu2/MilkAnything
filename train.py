import argparse
import sys
import torch
from torch.utils.data import Subset

from milkanything.nets.discriminator import Discriminator
from milkanything.nets.generator import Generator
from milkanything.utils.dataset import CycleGANDataset
from milkanything.utils.trainer import MilkAnythingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN with Custom Backbone Discriminator")

    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory (must contain "source" and "target" folders)')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='Directory to save checkpoints and training curves')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image resolution for training')

    parser.add_argument('--backbone', type=str, default='efficientnet_v2_l',
                        help='Torchvision backbone name for Discriminator')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights for the backbone (e.g., "DEFAULT")')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of total epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (adjust based on your GPU memory)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate for Adam optimizer')
    parser.add_argument('--lambda_cyc', type=float, default=2.0,
                        help='Weight for cycle consistency loss')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable Automatic Mixed Precision (AMP) training')
    parser.set_defaults(use_amp=True)

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type}")
    if device.type == 'cuda':
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    print("\n--- Loading Datasets ---")

    try:
        train_dataset_full = CycleGANDataset(root_dir=args.data_path, image_size=args.image_size, is_train=True)
        eval_dataset_full = CycleGANDataset(root_dir=args.data_path, image_size=args.image_size, is_train=False)

        total_size = len(train_dataset_full)
        train_size = int(0.65 * total_size)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(eval_dataset_full, eval_indices)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    print("\n--- Initializing Networks ---")
    print(f"Creating Generators (A2B & B2A)...")
    G_A2B = Generator()
    G_B2A = Generator()

    print(f"Creating Discriminators (D_A & D_B) with backbone: {args.backbone}...")

    D_A = Discriminator(backbone_name=args.backbone, weights=args.weights)
    D_B = Discriminator(backbone_name=args.backbone, weights=args.weights)

    print("\n--- Setting up Trainer ---")
    trainer = MilkAnythingTrainer(
        G_A2B=G_A2B,
        G_B2A=G_B2A,
        D_A=D_A,
        D_B=D_B,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lambda_cyc=args.lambda_cyc,
        device=device,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        use_amp=args.use_amp
    )

    print("\n--- Starting Training ---")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state is not guaranteed.")

if __name__ == "__main__":
    main()