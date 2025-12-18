import timm
import torch
from torch import nn


class VitFeatureExtractor(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224", pretrained=True, trainable_layers=None, dtype=torch.float32):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        if hasattr(self.model, "head"):
            self.model.head = nn.Identity()
        if trainable_layers is not None:
            for name, param in self.model.named_parameters():
                param.requires_grad = any([layer in name for layer in trainable_layers])
        print("Trainable parameters in Vision Encoder:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.to(dtype)

    def forward(self, img):
        if isinstance(img, list):
            img_tensor = torch.cat(img, dim=0)
        else:
            img_tensor = img
        features = self.model.forward_features(img_tensor)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:, :]
        return cls_token, patch_tokens

