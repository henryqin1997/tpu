export TPU_NAME=ziheng
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export MODEL_DIR=gs://ziheng_store
gsutil -m rm -R -f $MODEL_DIR/*
python resnet_main_batch_step —tpu=ziheng —log_step_count_steps 1 —resnet_depth 50 --train_batch_size 1024 —mode train —train_steps 2502 --data_dir=$DATA_DIR --model_dir=$MODEL_DIR
