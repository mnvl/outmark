#! /bin/bash
set -e

export PATH=/home/mel/anaconda3/bin:$PATH

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for device in "$@"
do
    port=`expr 17270 + $device`
    output="output${device}"
    log="train${device}.log"

    echo "Starting task: device = $device, port = $port, output = $output, log = $log"

    rm -rf $output
    mkdir $output

    CUDA_VISIBLE_DEVICES=$device python3 train.py \
			--verbose_feature_extractor=False \
			--image_server_rows_per_page=20 \
			--image_server_port=$port \
			--output=$output \
			--data_info_json /home/mel/datasets/LiTS-baked/info.json \
			--mode=train \
			--num_steps=500000 \
			--image_depth=1 \
			--image_width=224 \
			--image_height=224 \
			--batch_size=5 \
			--estimate_every_steps=25 \
			--validate_every_steps=10000 \
			>$log 2>&1 &
done

wait
