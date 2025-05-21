#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --job-name=train_merged30m
#SBATCH --output=./cluster_logs/train_merged30m.txt
#SBATCH --partition PARTITION_NAME
    # --train-data '/mnt/shared_38/data/xuboyu/datasets/DataCompDR-12M/train/{00000000..00001023}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-{0000..2812}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc12m_recap_wds/train/cc12m-train-{0000..2175}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc3m_recap_wds/train/cc3m-train-{0000..0575}.tar' \
    # --train-num-samples 37794565 \
    # --train-data '/mnt/shared_38/data/xuboyu/datasets/DataCompDR-12M/train/{00000000..00000001}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-{0000..0003}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc12m_recap_wds/train/cc12m-train-{0000..0003}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc3m_recap_wds/train/cc3m-train-{0000..0003}.tar' \
    # --train-num-samples 80000 \
    # --train-data '/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-0000.tar' \
    # --train-num-samples 5000 \
    # --train-data '/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-{0000..0059}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc12m_recap_wds/train/cc12m-train-{0000..0039}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc3m_recap_wds/train/cc3m-train-{0000..0039}.tar' \
    # --train-num-samples 1000000 \

export WANDB_DIR="/home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output"

params=(
    -m main 
    --logs-dir /home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output/logs/
    --model MobileCLIP-S0-S2 
    # --pretrained /home/xuboyu/Projects/CLIP/test_mobileclip/ml-mobileclip/outputs/checkpoints/mobileclip_s0_image_s2_text/mobileclip_s0_s2_combined.pt \ 
    --dataset-type webdataset
    --lr 8e-5 
    --warmup 2000 
    --epochs 3 
    --train-data '/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-{0000..0059}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc12m_recap_wds/train/cc12m-train-{0000..0039}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc3m_recap_wds/train/cc3m-train-{0000..0039}.tar' \
    --train-num-samples 1000000 
    --val-data 'coco' 
    --data-root-dir /mnt/shared_38/data/xuboyu/datasets 
    --batch-size 128 
    --precision amp 
    --workers 0 
    --save-frequency 1 
    --log-every-n-steps 200 
    --wd 0.5 
    --beta1 0.9 
    --beta2 0.98 
    --eps 1e-8 
    --use-imagecrop-aug 
    --global-crops-number 2 
    --local-crops-number 6 
    --crop-scale 0.4 
    --caption-sampling-mode textcrop 
    --num-sampled-captions 8 
    --momentum-teacher 0.99 
    --fix-momentum 
    --output-all 
    --attentional-pool 
    --cosmos 
    --report-to wandb 
    --wandb-project-name cosmos-mobileclip-s0-s2-merged28m 
    --accum-freq 1 
    --grad-checkpointing 
    # --lock 
    # --torchcompile
)
python "${params[@]}"