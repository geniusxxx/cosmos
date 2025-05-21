#!/bin/bash
RESOLUTION=256
FRAMEWORK=mobileclip
MODEL_PATH=/home/xuboyu/Projects/CLIP/test_mobileclip/cosmos/output/logs/2025_04_25-18_29_29-model_MobileCLIP-S2-lr_2e-05-b_64-j_0-p_amp-key_/checkpoints/epoch_1.pt
MODEL_ARCH=MobileCLIP-S2
PART=text #visual, text, both
VERIFY=True
DYNAMIC_AXES=False
NORMALIZE=False  # 添加normalize参数
MODEL_TYPE=student

MODEL_NAME=cosmos-merged30m

if [ ${MODEL_ARCH} = "MobileCLIP-S2" ]; then
    MODEL_NAME=${MODEL_NAME}-mobileclip-s2
fi

if [ ${PART} = "text" ]; then
    # 文本编码器不需要reparam和resolution参数
    MODEL_NAME=${MODEL_NAME}_${PART}
    EXPORT_ARG="--export-text"
elif [ ${PART} = "visual" ]; then
    # 视觉编码器需要包含resolution和reparam信息
    MODEL_NAME=${MODEL_NAME}_${RESOLUTION}_${PART}
    EXPORT_ARG="--export-image"
else  # both
    MODEL_NAME=${MODEL_NAME}_${RESOLUTION}_${PART}
    EXPORT_ARG="--export-all"
fi

if [ -n "${MODEL_PATH}" ] && [ ${MODEL_PATH,,} != "none" ]; then
    MODEL_NAME=${MODEL_NAME}_pretrained
    MODEL_PATH_ARG="--model-path ${MODEL_PATH}"
else
    MODEL_PATH_ARG=""
fi

if [ ${VERIFY,,} = "true" ]; then
    MODEL_NAME=${MODEL_NAME}_verify
    VERIFY_ARG="--verify"
else
    VERIFY_ARG=""
fi

if [ ${DYNAMIC_AXES,,} = "true" ]; then
    MODEL_NAME=${MODEL_NAME}_dynamic
    DYNAMIC_AXES_ARG="--dynamic-axes"
else
    DYNAMIC_AXES_ARG=""
fi

if [ ${MODEL_TYPE,,} = "teacher" ]; then
    MODEL_NAME=${MODEL_NAME}_${MODEL_TYPE}
    MODEL_TYPE_ARG="--model-type teacher"
elif [ ${MODEL_TYPE,,} = "student" ]; then
    MODEL_NAME=${MODEL_NAME}_${MODEL_TYPE}
    MODEL_TYPE_ARG="--model-type student"
else
    MODEL_TYPE_ARG=""
fi

# 添加normalize相关的命名逻辑
if [ ${NORMALIZE,,} = "true" ]; then
    MODEL_NAME=${MODEL_NAME}_norm
    NORMALIZE_ARG="--normalize true"
else
    NORMALIZE_ARG="--normalize false"
fi

params=(
    -m src.training.pytorch_to_onnx 
    --framework ${FRAMEWORK}
    --model-arch ${MODEL_ARCH}
    --model-path ${MODEL_PATH} 
    --output-path output/onnx/${MODEL_NAME}.onnx 
    --resolution ${RESOLUTION} 
    ${MODEL_PATH_ARG} 
    ${EXPORT_ARG} 
    ${VERIFY_ARG} 
    ${DYNAMIC_AXES_ARG}
    ${MODEL_TYPE_ARG}
    ${NORMALIZE_ARG}
)

# 执行训练命令
python "${params[@]}"