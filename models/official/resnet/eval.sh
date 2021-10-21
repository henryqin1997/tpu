export PYTHONPATH="$PYTHONPATH:/home/supercomputer_ai/tpu/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store/larsonecycle
export DATA_DIR=gs://imagenet2012_eu/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-8
python3 resnet_main.py \
    --tpu=${TPU_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --mode=eval \
    --config_file=configs/cloud/${ACCELERATOR_TYPE}.yaml
