import torch
import torch.nn.functional as F

class DINOv3UConnModel(torch.nn.Module):
    def __init__(self, num_classes=2, repo_dir="D:/dinov3/dinov3",
                 weights_path="../dinov3_vits16_pretrain_lvd1689m.pth",
                 freeze_backbone=True):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vits16",
            source="local",
            weights=weights_path,
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        return self.classifier(features.float())