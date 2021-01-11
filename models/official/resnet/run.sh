export PYTHONPATH="$PYTHONPATH:/path/to/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
#export DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
gsutil -m rm -R -f $MODEL_DIR/*
python3 official/resnet/resnet_main_batch_step.py —-tpu=ziheng --resnet_depth=50 --train_steps=200 —-mode=train --data_dir=$DATA_DIR --model_dir=$MODEL_DIR