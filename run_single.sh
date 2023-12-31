#!/bin/bash
#SBATCH --job-name=PCAMexp      # Job name
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=./logs/run_%j.log    # Standard output and error log, plik pojawia siÄ™ w miejscu odpalenia skryptu
#SBATCH --gres=gpu
#SBATCH --partition=short,experimental
#SBATCH --mem=64G
#SBATCH -A cause-lab

source ~/anaconda3/etc/profile.d/conda.sh

if [[ $1 == "pcammae" ]]
then
    conda activate PCam_context_MAE
else
    conda activate PCam_context
fi

python run_experiment_probs.py $1
