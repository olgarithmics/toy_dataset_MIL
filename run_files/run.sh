#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:05:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets/ColonCancer_results/vaegan_train.out
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets/ColonCancer_results/vaegan_train.err
#SBATCH --partition=gpu


module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate alma
cd /home/ofourkioti/Projects/toy_dataset_MIL/
python run.py --experiment_name test  --mode vaegan --k 3  --input_shape 27 27 3 --extention bmp --data colon --run 1 --vaegan_lr 0.0003
