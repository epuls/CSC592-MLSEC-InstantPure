import torch

class DINOv3ViTs16(torch.nn.Module):
    def __init__(
        self,
        repo_dir: str,
        weights_path: str,
        num_classes: int = 200,
        freeze_backbone: bool = False,
    ):
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
    



if __name__ == "__main__":
    # quick test to verify the model can be instantiated and run a forward pass
    model = DINOv3ViTs16(repo_dir='../dinov3/',
            weights_path='../dinov3/dinov3_vits16_pretrain_lvd1689m.pth',)
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    # print modules:
    for name, module in model.named_modules():
        print(f"{name}: {module}")