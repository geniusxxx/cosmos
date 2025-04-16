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

export WANDB_DIR="/home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output"

python -m main \
    --logs-dir /home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output/logs/ \
    --model ViT-B-32 \
    --pretrained /home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output/checkpoints/cosmos_vitb32_merged30m.pt \
    --dataset-type webdataset  \
    --lr 8e-5 \
    --warmup 2000 \
    --epochs 5  \
    --train-data '/mnt/shared_38/data/xuboyu/datasets/DataCompDR-12M/train/{00000000..00001023}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/yfcc15m_recap_wds/train/yfcc15m-train-{0000..2812}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc12m_recap_wds/train/cc12m-train-{0000..2175}.tar::/mnt/shared_38/data/xuboyu/datasets/DreamLIP28M/cc3m_recap_wds/train/cc3m-train-{0000..0575}.tar' \
    --train-num-samples 37794565 \
    --val-data 'coco' \
    --data-root-dir /mnt/shared_38/data/xuboyu/datasets \
    --batch-size 64 \
    --precision amp_bf16 \
    --workers 0 \
    --save-frequency 1 \
    --log-every-n-steps 200 \
    --wd 0.5 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-8 \
    --use-imagecrop-aug \
    --global-crops-number 2 \
    --local-crops-number 6 \
    --crop-scale 0.4 \
    --caption-sampling-mode textcrop \
    --num-sampled-captions 8 \
    --momentum-teacher 0.99 \
    --fix-momentum \
    --output-all \
    --attentional-pool \
    --cosmos \
    --report-to wandb \
    --wandb-project-name cosmos-vitb32-merged28m-datacompdr12m \
    --grad-checkpointing \
    --accum-freq 8 \
    # --torchcompile
