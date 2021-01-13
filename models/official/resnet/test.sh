export STORAGE_BUCKET=gs://ziheng_store
export MODEL_DIR=${STORAGE_BUCKET}/test
export PYTHONPATH="$PYTHONPATH:/tpu/models"
export TPU_NAME=ziheng
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-1024
gsutil -m rm -R -f $MODEL_DIR/*
python3 official/resnet/resnet_main.py \
    --tpu=${TPU_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --train_steps=500 \
    --config_file=configs/cloud/${ACCELERATOR_TYPE}.yaml