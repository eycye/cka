#!/bin/bash
#SBATCH --array=21,22,46,47,41
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/cka
##SBATCH --exclude=freddie,havoc,r4,r16,atlas,blaze,flaminio,steropes,como,ace
#SBATCH --nodelist=freddie
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet_avg_cka_%A_%a.out

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
method=average
# SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$init_lr
# /data/eliciaye/val_experiments/me-prune/cifar100/resnet34-5-0.05
# SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-$init_lr
SAVE=/data/eliciaye/val_experiments/me-prune/cifar100/resnet$depth-$idx-average_init$init_lr

echo $SAVE

python resnet_compare.py --checkpoint $SAVE --lr $init_lr --width_frac $width_frac --depth $depth --method $method
echo "All done."
