export PYTHONPATH="$PYTHONPATH:/path/to/models"
export TPU_NAME=ziheng
export MODEL_DIR=gs://ziheng_store/larslrrt/
export DATA_DIR=gs://imagenet2012/imagenet-2012-tfrecord/
gsutil -m rm -R -f $MODEL_DIR/*
python3 official/resnet/resnet_main_batch_step.py —-tpu=ziheng --resnet_depth=50 --data_dir=$DATA_DIR --model_dir=gs://ziheng_store/ --train_batch_size=32768 --train_steps=200 --mode=train
python3 official/resnet/resnet_main_batch_step.py —-tpu=ziheng --enable_lars --resnet_depth=50 --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --train_batch_size=32768 --train_steps=200 --mode=train