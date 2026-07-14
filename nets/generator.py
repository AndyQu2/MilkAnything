import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels):
        super(TransformerBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：x + f(x)
        return x + self.conv_block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1):
        super(DecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bn = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_encoder, num_transformers, num_decoder):
        super(Generator, self).__init__()

        self.num_encoder = num_encoder
        self.num_transformers = num_transformers
        self.num_decoder = num_decoder

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            EncoderBlock(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0),
        )
        channels = self._build_encoder()

        self.transformers = nn.Sequential()
        self._build_transformer(channels)

        self.decoder = nn.Sequential()
        self._build_decoder(channels)

    def _build_encoder(self):
        channels = 64
        for _ in range(self.num_encoder - 1):
            self.encoder.append(EncoderBlock(in_channels=channels, out_channels=channels * 2,
                                             kernel_size=3, stride=2, padding=1))
            channels = channels * 2
        return channels

    def _build_transformer(self, channels):
        for _ in range(self.num_transformers):
            self.transformers.append(TransformerBlock(channels=channels))

    def _build_decoder(self, channels):
        for _ in range(self.num_decoder - 1):
            self.decoder.append(DecoderBlock(in_channels=channels, out_channels=channels // 2,
                                             kernel_size=3, stride=2, padding=1, output_padding=1))
            channels = channels // 2

        self.decoder.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.transformers(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = Generator(num_encoder=3, num_transformers=9, num_decoder=3)

    print("Model Summary:")
    print(model)

    dummy_input = torch.randn(1, 3, 512, 512)
    dummy_output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")