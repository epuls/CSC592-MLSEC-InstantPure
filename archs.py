# modify on top of https://github.com/xavihart/Diff-PGD

from dataset import get_normalize_layer
import torchvision
from transformers import BeitForImageClassification
import torch
from cub_dino_v3_model import DINOv3ViTs16
from uconn_ballots_dino_v3_model import DINOv3UConnModel


def get_archs(arch, dataset='imagenet'):
    if dataset == 'imagenet':
        if arch == 'resnet50':
            model = torchvision.models.resnet50(weights="DEFAULT")
        
        elif arch == 'resnet152':
            model = torchvision.models.resnet152(weights="DEFAULT")
            
        elif arch == 'wrn50':
            model = torchvision.models.wide_resnet50_2(weights="DEFAULT")

        elif arch == 'vit':
            model = torchvision.models.vit_b_16(weights='DEFAULT')

        elif arch == 'swin_b':
            model = torchvision.models.swin_b(weights='DEFAULT')

        elif arch == 'convnext_b':
            model = torchvision.models.convnext_base(weights='DEFAULT')
    elif dataset == "cub200":
        model = DINOv3ViTs16(
            repo_dir='../dinov3/',
            weights_path='../dinov3/dinov3_vits16_pretrain_lvd1689m.pth',
            num_classes=200,
            freeze_backbone=True,
        )

        weights = torch.load('../cub_dinov3_vits16.pt', map_location='cuda')

        model.load_state_dict(weights['model_state_dict'])
        model.eval()
    elif dataset == "uconn":
        model = DINOv3UConnModel(num_classes=2, 
                                 repo_dir='../dinov3/', 
                                 weights_path='../dinov3/dinov3_vits16_pretrain_lvd1689m.pth', 
                                 freeze_backbone=True
                                 )
        weights = torch.load('../uconn_dinov3_vits16.pt', map_location='cuda')
        model.load_state_dict(weights['model_state_dict'])
        model.eval()
        
    normalize_layer = get_normalize_layer(dataset)
    
    return torch.nn.Sequential(normalize_layer, model)

    
