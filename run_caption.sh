#!/bin/bash

MODEL_PATH="/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/68156fd997cdc9f710620466735af49862bb81f6/"
# MODEL_NAME="/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct-1M/snapshots/620fad32de7bdd2293b3d99b39eba2fe63e97438/"
python run_video_caption.py \
        --model_name "$MODEL_PATH" 
    
echo "All processing completed!" 
