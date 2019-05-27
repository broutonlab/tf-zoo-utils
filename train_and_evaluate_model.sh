#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 model_directory"
    echo "Trains and evaluates the model. Saves checkpoints into \"<model_directory>/train\" and"
    echo "\"<model_directory>/train/all_checkpoints\""
    exit 0
fi

save_checkpoints() {
    mkdir -p $1/train/all_checkpoints
    while :
    do
        cp -f $1/train/model.ckpt* $1/train/all_checkpoints/ &> /dev/null
        sleep 5m
    done
}

save_checkpoints $1 &
save_checkpoints_pid=$!

trap "kill $save_checkpoints_pid" SIGINT SIGTERM EXIT

PIPELINE_CONFIG_PATH=$1/pipeline.config
MODEL_DIR=$1/train
NUM_TRAIN_STEPS=`grep -Po 'num_steps:\s+\K([0-9]+)' $PIPELINE_CONFIG_PATH`
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 $PATH_TO_TF_MODELS_RESEARCH/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS}
    #--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES
    #--alsologtostderr Is not supported
