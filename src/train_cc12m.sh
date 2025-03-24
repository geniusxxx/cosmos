#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --job-name=train_cc12m
#SBATCH --output=./cluster_logs/train_cc12m.txt
#SBATCH --partition PARTITION_NAME

source cosmos_env/bin/activate
cd cosmos/src

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun env -u CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=4 \
    --nnode=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    -m main \
    --logs-dir ./logs/ \
    --model ViT-B-16 \
    --dataset-type webdataset  \
    --lr 5e-4 \
    --warmup 2000 \
    --epochs 32  \
    --train-data 'datasets/cc12m_recap/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10010225 \
    --val-data 'coco' \
    --data-root-dir directory/to/coco/ \
    --batch-size 32 \
    --precision amp \
    --workers 16 \
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
    --cosmos
