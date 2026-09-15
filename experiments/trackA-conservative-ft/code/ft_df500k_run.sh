#!/bin/bash
cd /workspace/ayb/Pocket2Mol
export PYTHONPATH=/workspace/ayb/Pocket2Mol:$PYTHONPATH
CUDA_VISIBLE_DEVICES=7 /workspace/ayb/miniconda3/envs/zatom310/bin/python -u train_resume.py --config configs/train_trackA_ft.yml --logdir /workspace/ayb/experiments/dfe-unified-trackA/ft_df500k --resume logs/checkpoints/500000.pt > /workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/train.log 2>&1
touch /workspace/ayb/experiments/dfe-unified-trackA/ft_df500k/done.flag
