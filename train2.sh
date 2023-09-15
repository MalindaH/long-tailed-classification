#!/bin/bash
#SBATCH --job-name=my_moe75         # create a short name for your job
#SBATCH --constraint=gpu80
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=lh9998@princeton.edu
## #SBATCH -e logs/slurm-%j.err
## #SBATCH -o logs/slurm-%j.out


# module purge
# # module load anaconda3/2022.5
# conda deactivate
# conda activate py3.9-torch
# cd ~/exp1/scripts3/my/


# python train.py --config="configs/config_isic3.3.json" --resume="torch_model_checkpoints/2023_04_30_09_15_00/testnet_epoch47.pt"
# python train.py --config="configs/config_isic4.2.json"
# python train.py --config="configs/config_isic7.1.json" --resume="torch_model_checkpoints/2023_05_05_22_42_27/testnet_epoch29.pt"
python train.py --config="configs/config_isic7.5.json" --resume="torch_model_checkpoints/2023_05_08_15_36_04/testnet_epoch29.pt"