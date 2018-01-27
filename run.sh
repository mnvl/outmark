#! /bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 ./train.py --settings LCTSC --data_info_json ~/datasets/LCTSC-baked/info.json --image_server_port=7000 --image_depth=1 --image_width=224 --image_height=224 --verbose_feature_extractor=False --override_background=2000 >train.log 2>train.log
