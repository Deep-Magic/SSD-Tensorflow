DATASET_DIR=/home/walter/Documents/my_git/pybot/util/training_data/rocks/
OUTPUT_DIR=/home/walter/Documents/others_git/tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
