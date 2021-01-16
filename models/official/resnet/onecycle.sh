export PYTHONPATH="$PYTHONPATH:/home/supercomputer_ai/tpu/models"
export TPU_NAME=v3-1024
export MODEL_DIR=gs://ziheng_store/larsonecycle
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-1024
gsutil -m rm -R -f $MODEL_DIR/*
python3 resnet_main_onecycle.py \
    --tpu=${TPU_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --config_file=configs/cloud/${ACCELERATOR_TYPE}.yaml
