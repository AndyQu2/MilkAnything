import os

import torch
from PIL import Image
from torchvision import transforms

from milkanything.nets.generator import Generator


class MilkAnythingInference:
    def __init__(self, checkpoint_path: str, direction: str = 'A2B', image_size: int = 256, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.direction = direction.upper()

        print(f"[{self.direction}] Initializing Generator...")
        if self.direction == 'A2B':
            self.netG = Generator()
        elif self.direction == 'B2A':
            self.netG = Generator()
        else:
            raise ValueError("Direction must be either 'A2B' or 'B2A'")

        self._load_weights(checkpoint_path)
        self.netG.to(self.device)
        self.netG.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 映射到 [-1, 1]
        ])

    def _load_weights(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict_key = 'G_A2B_state_dict' if self.direction == 'A2B' else 'G_B2A_state_dict'

        if state_dict_key in checkpoint:
            self.netG.load_state_dict(checkpoint[state_dict_key])
        else:
            try:
                self.netG.load_state_dict(checkpoint)
            except RuntimeError as e:
                raise KeyError(f"Could not find {state_dict_key} in checkpoint. "
                               f"Please check your checkpoint keys. Error: {e}")
        print("Generator weights loaded successfully.")

    @torch.no_grad()
    def predict(self, img_path: str) -> tuple[Image.Image, Image.Image]:
        orig_img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(orig_img).unsqueeze(0).to(self.device)

        fake_tensor = self.netG(img_tensor)

        fake_tensor = fake_tensor.squeeze(0).cpu().float()
        fake_tensor = (fake_tensor + 1.0) / 2.0  # 映射到 [0, 1]
        fake_tensor = torch.clamp(fake_tensor, 0.0, 1.0)

        to_pil = transforms.ToPILImage()
        translated_img = to_pil(fake_tensor)

        translated_img = translated_img.resize(orig_img.size, Image.BICUBIC)

        return orig_img, translated_img