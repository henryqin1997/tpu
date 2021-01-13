export PYTHONPATH="$PYTHONPATH:~/tpu/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store/lars
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
export ACCELERATOR_TYPE=v3-1024
gsutil -m rm -R -f $MODEL_DIR/*
#python3 official/resnet/resnet_main_batch_step.py --tpu=${TPU_NAME} \
#  --resnet_depth=50 --data_dir=$DATA_DIR \
#  --model_dir=gs://ziheng_store/ --train_batch_size=32768 \
#   --train_steps=200 --mode=train --config_file=official/resnet/configs/cloud/${ACCELERATOR_TYPE}.yaml
python3 official/resnet/resnet_main_batch_step.py —-tpu=ziheng \
--data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR --train_batch_size=32768 --train_steps=200 \
   --mode=train --config_file=official/resnet/configs/cloud/${ACCELERATOR_TYPE}.yaml