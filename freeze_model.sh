#!/bin/bash

if [[ $# -ne 2 && $# -ne 3 ]]; then
    echo "Usage: $0 model_directory checkpoint_number [output_path]"
    echo "Freezes checkpoint \"<model_directory>/train/model.ckpt-<checkpoint_number>\""
    echo "or, if that one doesn't exist, then"
    echo "\"<model_directory>/train/all_checkpoints/model.ckpt-<checkpoint_number>\". Saves"
    echo "the frozen graph into \"<output_path>\" or, if output_path is not specified, into"
    echo "\"<model_directory>/frozen_model_<date_and_time>\"."
    exit 0
fi

PIPELINE_CONFIG_PATH=$1/pipeline.config
CHECKPOINT_PATH=$1/train/model.ckpt-$2
if [ ! -f "${CHECKPOINT_PATH}.index" ]; then
    CHECKPOINT_PATH=$1/train/all_checkpoints/model.ckpt-$2
    if [ ! -f "${CHECKPOINT_PATH}.index" ]; then
        echo "Checkpoint $@ doesn't exist"
        exit 1
    fi
fi

if [ $# -eq 2 ]
then
    OUTPUT_PATH="$1/frozen_model_$(date +%F-%H-%M-%S)"
else
    OUTPUT_PATH=$3
fi

python3 $PATH_TO_TF_MODELS_RESEARCH/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${CHECKPOINT_PATH} \
    --output_directory=${OUTPUT_PATH}
