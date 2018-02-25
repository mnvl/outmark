#! /bin/sh

dataset=$1
instance=$2

verbose=True
port=`expr 7000 + $instance`

echo "*** instance=$instance"
echo "*** dataset=$dataset"
echo "*** port=$port"

export CUDA_VISIBLE_DEVICES=$instance

mkdir output

case $dataset in
    LCTSC)
        python3 ./train.py \
                --settings LCTSC \
                --data_info_json ~/datasets/LCTSC-baked/info.json \
                --image_server_port=$port \
                --batch_size 4 --image_depth=1 --image_width=448 --image_height=448 \
                --verbose_feature_extractor=$verbose \
                --num_steps=1000000 \
                --validate_every_steps=5000 \
                >output/train.log 2>&1
        ;;

    Tissue)
        python3 ./train.py \
                --settings Tissue \
                --data_info_json ~/datasets/tissue-baked/info.json \
                --image_server_port=$port \
                -batch_size 4 --image_depth=1 --image_width=448 --image_height=448 \
                --verbose_feature_extractor=$verbose \
                --num_steps=1000000 \
                --validate_every_steps=5000 \
                >output/train.log 2>&1
        ;;

    *)
        echo "*** usage: ./run.sh dataset instance"
        exit 1
esac
