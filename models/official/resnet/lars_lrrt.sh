export PYTHONPATH="$PYTHONPATH:/usr/share/tpu/models"
export TPU_NAME=zihenglars
export MODEL_DIR=gs://ziheng_store/lars
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-1024
gsutil -m rm -R -f $MODEL_DIR/*
python3 resnet_main_batch_step.py —-tpu=ziheng \
    --tpu=${TPU_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --train_steps=200 \
    --config_file=configs/cloud/${ACCELERATOR_TYPE}.yaml