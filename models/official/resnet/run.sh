export PYTHONPATH="$PYTHONPATH:/path/to/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
#export DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
gsutil -m rm -R -f $MODEL_DIR/*
python3 official/resnet/resnet_main_batch_step.py —tpu=ziheng —log_step_count_steps 1 —resnet_depth 50 --train_batch_size 1024 —mode train —train_steps 2502 --data_dir=$DATA_DIR --model_dir=$MODEL_DIR
