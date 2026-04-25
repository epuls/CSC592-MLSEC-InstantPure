import torch
import torch.nn.functional as F

class DINOv3UConnModel(torch.nn.Module):
    def __init__(
            self, 
            num_classes=2, 
            repo_dir="D:/dinov3/dinov3",
            weights_path="../dinov3_vits16_pretrain_lvd1689m.pth",
            freeze_backbone=True):
        super().__init__()

        super().__init__()

        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vits16",
            source="local",
            weights=weights_path,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        logits = self.classifier(outputs)
        return logits