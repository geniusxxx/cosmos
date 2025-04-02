# COSMOS models
# --model ViT-B-16 
# --huggingface-model-name [cosmos_vitb16_cc3m, cosmos_vitb16_cc12m, cosmos_vitb16_yfcc15m, cosmos_vitb16_merged30m, cosmos_vitb16_pixelprose]
# --model ViT-B-32 
# --huggingface-model-name [cosmos_vitb32_cc3m, cosmos_vitb32_cc12m, cosmos_vitb32_yfcc15m, cosmos_vitb32_merged30m, cosmos_vitb32_pixelprose]
torchrun --nproc_per_node 1 -m main \
    --model ViT-B-16 \
    --huggingface-repo-name sankim2/cosmos \
    --pretrained /home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output/checkpoints/cosmos_vitb32_merged30m.pt \
    --val-data retrieval  \
    --data-root-dir /mnt/shared_8_common/Public_Datasets \
    --batch-size 256 \
    --workers 16 \
    --output-all \
    --attentional-pool  \
    --cosmos \

# OpenCLIP models 
# --model ViT-B-16 --pretrained [laion400m_e32, datacomp_xl_s13b_b90k, laion2b_s34b_b88k]
# --model ViT-B-32 --pretrained [laion400m_e32, datacomp_xl_s13b_b90k, laion2b_s34b_b79k]
# torchrun --nproc_per_node 1 -m main \
#     --model ViT-B-16 \
#     --pretrained laion400m_e32 \
#     --val-data retrieval  \
#     --data-root-dir /directory/to/your/coco/and/flickr30k/ \
#     --batch-size 256 \
#     --workers 16 \