#!/bin/bash
#SBATCH --array=1-5
#SBATCH -p rise # partition (queue)
#SBATCH -D /home/eecs/eliciaye/cka
##BATCH --exclude=havoc,r4,r16,atlas,blaze
#SBATCH --nodelist=zanino
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=1 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -o resnet18_25_cka%A_%a.out

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
cd /home/eecs/eliciaye/cka

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p resnet_new.txt)
SAVE=$(echo $cfg | cut -f 1 -d ' ')
init_lr=$(echo $cfg | cut -f 2 -d ' ')
width_frac=0.25
depth=18
method=ww
echo $SAVE

python resnet_compare_new.py --checkpoint $SAVE --lr $init_lr --width_frac $width_frac --depth $depth --method $method
echo "All done."
