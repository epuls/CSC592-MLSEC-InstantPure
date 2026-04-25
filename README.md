# Debugging WIP
## Env Vars
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./logs/OSCP"

## Train Commands
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
    --max_train_steps=10000 \
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

## Test Commands
python test.py --model="LCM" --output_dir="output/" --num_validation_set=1000 --lora_input_dir="logs/OSCP/unet_lora/" --strength=0.2 --num_inference_step=5 --device="cuda" --attack_method="Linf_pgd"

python test.py --model="LCM" --output_dir="output/" --num_validation_set=1000 --lora_input_dir="logs/OSCP/unet_lora/" --strength=0.2 --num_inference_step=5 --device="cuda" --attack_method="AutoAttack"

## TMP Advertorch patch
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    else:
        for elem in x:
            zero_gradients(elem)



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
