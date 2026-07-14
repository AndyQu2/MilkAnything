import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CustomizedDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train

        self.image_paths = self._load_images(root_dir)
        self.total_size = len(self.image_paths)

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
    def _load_images(root_dir):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp',
                            '.HEIC', '.heic', '.JPG', '.JPEG', '.JPEG', '.PNG', '.PNG'}
        paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    paths.append(os.path.join(root, file))

        if len(paths) == 0:
            raise RuntimeError(f"Error: No valid images found in folder: '{root_dir}'")
        return paths

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        label = torch.tensor([0.0, 1.0], dtype=torch.float32)

        return img_tensor, label