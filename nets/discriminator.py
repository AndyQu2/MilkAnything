import torch
from torch import nn
from torchvision.models import get_model


class Discriminator(nn.Module):
    def __init__(self, backbone_name: str = 'efficientnet_v2_l', weight = None, dropout_rate: float = 0.2):
        super(Discriminator, self).__init__()

        self.dropout_rate = dropout_rate
        self.backbone = get_model(backbone_name, weights=weight)
        self._modify_classifier()

    def _modify_classifier(self):
        child_module = list(self.backbone.named_children())

        last_layer_name, last_layer_module = child_module[-1]
        if isinstance(last_layer_module, nn.Sequential):
            in_features = None
            for sub_module in last_layer_module.modules():
                if hasattr(sub_module, 'in_features'):
                    in_features = sub_module.in_features
                    break
        else:
            in_features = last_layer_module.in_features

        if in_features is None:
            raise ValueError('Can not determine in_features automatically.')

        setattr(self.backbone, last_layer_name, nn.Sequential(
            nn.Linear(int(in_features), int(in_features) // 2, bias=False),
            nn.BatchNorm1d(int(in_features) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),

            nn.Linear(int(in_features) // 2, int(in_features) // 4, bias=False),
            nn.BatchNorm1d(int(in_features) // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(int(in_features) // 4, 2),
            nn.Sigmoid()
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x