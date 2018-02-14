#! /bin/sh

export CUDA_VISIBLE_DEVICES=2
mkdir output
python3 ./train.py --settings LCTSC --read_model output/checkpoint_139999. --data_info_json ~/datasets/LCTSC-baked/info.json --image_server_port=7002 --batch_size 4 --image_depth=1 --image_width=448 --image_height=448 --verbose_feature_extractor=False --override_background=2000 --num_steps=1000000 --validate_every_steps=5000 >train.log 2>train.log
