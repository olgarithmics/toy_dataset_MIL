#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=70:05:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets_util/ColonCancer_results/info_gan_layer.out
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets_util/ColonCancer_results/info_gan_layer.err
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate alma
cd /home/ofourkioti/Projects/toy_dataset_MIL/
python run.py --experiment_name info_gan --mode vaegan  --k 3 --input_shape 27 27 3 --extention bmp --data colon  --data_path ColonCancer  --vaegan_save_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets/infogan_weights