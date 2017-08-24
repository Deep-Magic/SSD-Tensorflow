DATASET_DIR=/home/walter/Documents/others_git/tfrecords/
TRAIN_DIR=/home/walter/Documents/others_git/SSD-Tensorflow/logs/
CHECKPOINT_PATH=/home/walter/Documents/others_git/SSD-Tensorflow/logs/model.ckpt-20095
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=10
