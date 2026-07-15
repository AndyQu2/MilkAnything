import torch
from torch.utils.data import Subset

from nets.discriminator import Discriminator
from utils.dataset import DiscriminatorDataset
from utils.discriminator_trainer import DiscriminatorTrainer


def main():
    model = Discriminator(backbone_name="efficientnet_v2_l", weights=None, dropout_rate=0.2)

    generator = torch.Generator().manual_seed(42)
    train_dataset_full = DiscriminatorDataset("datasets", image_size=512, is_train=True)
    eval_dataset_full = DiscriminatorDataset("datasets", image_size=512, is_train=False)

    total_size = len(train_dataset_full)
    train_size = int(0.8 * total_size)
    eval_size = total_size - train_size

    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    eval_dataset = Subset(eval_dataset_full, eval_indices)

    print(f"Total images: {total_size}")
    print(f"Train subset size (Augmented): {len(train_dataset)}")
    print(f"Eval subset size (Standard): {len(eval_dataset)}")

    trainer = DiscriminatorTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=16,
        num_epochs=200,
        learning_rate=1e-3,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        patience=50,
        num_workers=8,
        save_dir='./output',
        auto_save=True,
        enable_early_stop=True,
        use_amp=True,
        dtype=torch.bfloat16
    )
    trainer.train()

if __name__ == "__main__":
    main()