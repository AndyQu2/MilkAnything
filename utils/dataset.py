import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DiscriminatorDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        super(DiscriminatorDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train

        self.source_dir = os.path.join(root_dir, 'source')
        self.target_dir = os.path.join(root_dir, 'target')

        self.samples = []

        if os.path.exists(self.source_dir):
            source_paths = self._load_images(self.source_dir)
            for path in source_paths:
                self.samples.append((path, [1.0, 0.0]))
        else:
            print(f"Warning: Source directory '{self.source_dir}' does not exist.")

        if os.path.exists(self.target_dir):
            target_paths = self._load_images(self.target_dir)
            for path in target_paths:
                self.samples.append((path, [0.0, 1.0]))
        else:
            print(f"Warning: Target directory '{self.target_dir}' does not exist.")

        self.total_size = len(self.samples)
        if self.total_size == 0:
            raise RuntimeError(f"Error: No valid images found in source or target folders under: '{root_dir}'")

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    @staticmethod
    def _load_images(dir_path):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp',
                            '.heic', '.jpg', '.jpeg', '.png'}
        paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    paths.append(os.path.join(root, file))
        return paths

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        img_path, label_list = self.samples[index]

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        # 转换为 PyTorch FloatTensor
        label = torch.tensor(label_list, dtype=torch.float32)

        return img_tensor, label