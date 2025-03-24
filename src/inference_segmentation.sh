# COSMOS models
# --model ViT-B-16 
# --huggingface-model-name [cosmos_vitb16_cc3m, cosmos_vitb16_cc12m, cosmos_vitb16_yfcc15m, cosmos_vitb16_merged30m, cosmos_vitb16_pixelprose]
# --model ViT-B-32 
# --huggingface-model-name [cosmos_vitb32_cc3m, cosmos_vitb32_cc12m, cosmos_vitb32_yfcc15m, cosmos_vitb32_merged30m, cosmos_vitb32_pixelprose]
# --seg-w-background : segmentation with background benchmarks
# --use-csa : using Correlative Self-Attention (CSA) block from SCLIP
torchrun --nproc_per_node 1 -m seg_eval.py  \
    --model ViT-B-16 \
    --huggingface-repo-name sankim2/cosmos \
    --huggingface-model-name cosmos_vitb16_merged30m.pt \
    --batch-size 256 \
    --workers 16 \
    --output-all \
    --attentional-pool  \
    --cosmos \
    --seg-w-background \
    --use-csa 

# OpenCLIP models 
# --model ViT-B-16 --pretrained [laion400m_e32, datacomp_xl_s13b_b90k, laion2b_s34b_b88k]
# --model ViT-B-32 --pretrained [laion400m_e32, datacomp_xl_s13b_b90k, laion2b_s34b_b79k]
torchrun --nproc_per_node 1 -m seg_eval.py  \
    --model ViT-B-16 \
    --pretrained laion400m_e32 \
    --batch-size 256 \
    --workers 16 \