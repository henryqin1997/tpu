export PYTHONPATH="$PYTHONPATH:~/tpu/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store/larssmall
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-8
gsutil -m rm -R -f $MODEL_DIR/*
python3 resnet_main_batch_step.py \
    --enable_lars \
    --tpu=${TPU_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --train_steps=400 \
    --config_file=configs/cloud/${ACCELERATOR_TYPE}.yaml