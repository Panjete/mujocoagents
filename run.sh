#!/bin/bash

#python scripts/train_agent.py --env_name Hopper-v4 --exp_name RL --video_log_freq 20 --scalar_log_freq 5 --no_gpu
echo "TRAIN 1 DONE"
#python scripts/train_agent.py --env_name HalfCheetah-v4 --exp_name RL --video_log_freq 20 --scalar_log_freq 5 --no_gpu
echo "TRAIN 2 DONE"
python scripts/train_agent.py --env_name Hopper-v4 --exp_name RL --video_log_freq 20 --scalar_log_freq 5 --no_gpu
echo "TRAIN 3 DONE"
python scripts/train_agent.py --env_name HalfCheetah-v4 --exp_name RL --video_log_freq 20 --scalar_log_freq 5 --no_gpu
echo "TRAIN 4 DONE"
python scripts/train_agent.py --env_name Ant-v4 --exp_name RL --video_log_freq 20 --scalar_log_freq 5 --no_gpu
echo "TRAIN 5 DONE"