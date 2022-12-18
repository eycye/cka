#!/bin/bash
##SBATCH --array=1-8,10-11,13,15-16,19,38 (steropes)
#SBATCH --array=9,21,24,26-27,30,32,39,42-43,45,49
##SBATCH --array=12,20,23,25,28-29,33,35,37,46,50 (como)
##SBATCH --array=14,17-18,22,31,34,36,40-41,44,47-48 (pavia)
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/cka
##SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,steropes,como,ace
#SBATCH --nodelist=zanino
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_baseline_cka_%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
cd /home/eecs/eliciaye/cka

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p resnet_config.txt)
idx=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
width_frac=$(echo $cfg | cut -f 3 -d ' ')
init_lr=$(echo $cfg | cut -f 4 -d ' ')

# /data/eliciaye/val_experiments/me-prune/cifar100/resnet34-0-0.025-baseline
# SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$init_lr
SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$init_lr-baseline
echo $SAVE

python resnet_compare.py --checkpoint $SAVE --lr $init_lr --width_frac $width_frac --depth $depth
echo "All done."
