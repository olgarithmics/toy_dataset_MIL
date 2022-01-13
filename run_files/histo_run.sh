#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:05:00
#SBATCH --output=/home/ofourkioti/Projects/toy_dataset_MIL/results/average_layer%j.out
#SBATCH --error=/home/ofourkioti/Projects/toy_dataset_MIL/results/average_layer%j.err
#SBATCH --partition=gpu


module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate histo_env
cd /home/ofourkioti/Projects/toy_dataset_MIL/
python run.py --experiment_name  k_1_euclidean --mode euclidean --k 1 --data_path ColonCancer --input_shape 27 27 3 --extention bmp --data colon

