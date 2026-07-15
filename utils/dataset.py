import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CycleGANDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        super(CycleGANDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train

        self.source_dir = os.path.join(root_dir, 'source')
        self.target_dir = os.path.join(root_dir, 'target')

        self.source_paths = self._load_images(self.source_dir) if os.path.exists(self.source_dir) else []
        self.target_paths = self._load_images(self.target_dir) if os.path.exists(self.target_dir) else []

        if len(self.source_paths) == 0 or len(self.target_paths) == 0:
            raise RuntimeError(f"Error: Need both source and target images under: '{root_dir}'")

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}
        paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    paths.append(os.path.join(root, file))
        return sorted(paths)

    def __len__(self):
        return max(len(self.source_paths), len(self.target_paths))

    def __getitem__(self, index):
        path_A = self.source_paths[index % len(self.source_paths)]
        path_B = random.choice(self.target_paths)

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        item_A = self.transform(img_A)
        item_B = self.transform(img_B)

        return {"A": item_A, "B": item_B}