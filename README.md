# How to run

## Setup
### env vars
These are default env vars if you follow the folder structure in below instructions. 
```
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./logs/OSCP"
export DM_CKPT="../Diff-PGD/ckpt/256x256_diffusion_uncond.pt"
export DINOV3_REPO="../dinov3"
export UCONN_DN_CKPT="./uconn_dinov3_vits16.pt"
```

### Requirements
**NOTE** Original Repo requirements.txt has too many conflicting/impossible dependencies. Versions removed to get original code functional

1. AutoAttack: `pip install git+https://github.com/fra31/auto-attack.git`
2. requirements.txt `pip install -r requirements.txt`
3. If using pytorch > 1.9 you will likely need to patch AdverTools' use of zero_gradients. Replace calls with below function:
```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    else:
        for elem in x:
            zero_gradients(elem)
```

### 1. Diff-PGD
1. Download the Diff-PGD checkpoint at: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt (repository: https://github.com/xavihart/Diff-PGD)

### 2. DINOv3 Backbone
1. Download DINOv3 weights at: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ (ViT-S/16 distilled)
2. Clone their repository: https://github.com/facebookresearch/dinov3 into the same parent folder as this repo
3. Place the downloaded DINOv3 weights in the cloned dino repo folder

### 3. UConn Voter Center Dataset
1. Download Voting Bubbles with Marginal Marks v2.2.0: https://zenodo.org/records/19189220 
2. Place in this repository's folder, so you should have: ../this_repo/uconn_voter_center_v2_2/FINALDATASETV3...
3. Download split_Combined_Grayscale.txt from: https://drive.google.com/drive/folders/1H3iXL8TUGtMu9xxJibnscobSGLULEBh1?usp=drive_link
4. Place .txt file in FINALDATASETV3 folder in uconn_voter_center_v2_2

### 4. LoRA Checkpoint
If only running tests against LoRA Checkpoint:
1. Download weights (uconn_dinov3_vits16.pt) from: https://drive.google.com/drive/folders/1H3iXL8TUGtMu9xxJibnscobSGLULEBh1?usp=drive_link
2. Place in OUTPUT_DIR, ensuring folder structure matches

## Experiment
### Annotated Code Presentations
For demonstration purposes, code has been extracted from original train.py/test.py and placed in jupyter notebooks, prefixed with exp_NAME.ipynb

```
exp_uconn_trainer.ipynb     Classifier training
exp_uconn_eval.ipynb        Classifier additional analysis
exp_train_lora.ipynb        LoRA (OSCP) Train
exp_test.ipynb              Purification Defense Tests (autoattack + PGD)
exp_test_diff_pgd.ipynb     Purification Defense Tests (DiffPGD)
exp_analysis.ipnb           Results analysis

```

### LoRA Training Without accelerate
After installing all of the above run below command. **NOTE** Ensure env vars are set!
```
python train_lora.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --lora_rank=64 \
    --lmd=0.001 \
    --N=2 \
    --learning_rate=1e-4 \
    --loss_type="l2" \
    --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=40000 \
    --dataloader_num_workers=8 \
    --checkpointing_steps=2000 \
    --checkpoints_total_limit=10 \
    --train_batch_size=16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --lr_scheduler="constant_with_warmup" \
    --resume_from_checkpoint=latest \
    --report_to="tensorboard" \
    --seed=3407
```

### LoRA Training With accelerate (not recommended but how original paper trained)
After installing all of the above run below command. **NOTE** Ensure env vars are set!

```
accelerate launch train_lora.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --lora_rank=64 \
    --lmd=0.001 \
    --N=2 \
    --learning_rate=1e-4 --loss_type="l2" --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=40000 \
    --dataloader_num_workers=8 \
    --checkpointing_steps=2000 --checkpoints_total_limit=10 \
    --train_batch_size=16 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --lr_scheduler="constant_with_warmup" \
    --resume_from_checkpoint=latest \
    --report_to="tensorboard" \
    --seed=3407

```

### Defense Evaluation
1. Run `run_uconn_tests.sh` for a battery of attacks/purification. 
2. Optionally, run exp_test.ipynb or exp_test_diff_pgd.ipynb instead





# Instant Adversarial Purification with Adversarial Consistency Distillation (CVPR2025)
## [Paper (arXiv)](https://arxiv.org/abs/2408.17064) 

Official Implementation for CVPR 2025 paper Instant Adversarial Purification with Adversarial Consistency Distillation.

![teaser](asset/teaser.png)


**Stable Diffusion:** Our model is developed by distilling Stable Diffusion v1.5 with a special LCM LoRA objective.
## Training objective
![obj](asset/pipeline_l.png)

---
## Train
Once you have prepared the data, you can train the model using the following command. 

```
bash train_lora.sh
```
---
## Evaluation
Evaluation code for ImageNet is provided.

```
bash test.sh
```
---
## Purification pipeline
![more](asset/pipeline_r.png)

All code run on NVIDIA L40 with cuda 12.4

## Checkpoint

https://drive.google.com/drive/folders/1bemjyZ4NyTeh9-0awkYdJ1jKH712LJ5y?usp=share_link

## Citation
Consider cite us if you find our paper is useful in your research :).
```
@InProceedings{Lei_2025_CVPR,
    author    = {Lei, Chun Tong and Yam, Hon Ming and Guo, Zhongliang and Qian, Yifei and Lau, Chun Pong},
    title     = {Instant Adversarial Purification with Adversarial Consistency Distillation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {24331-24340}
}
```
