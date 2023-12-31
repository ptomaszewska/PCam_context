#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n PCam_context_MAE python=3.8 -y
conda activate PCam_context_MAE

conda install -y pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

conda install -y pandas matplotlib seaborn scikit-learn scipy

# for transformers
pip install timm==0.4.12
pip install ml-collections

pip install h5py
pip install torchmetrics 
pip install --upgrade torchaudio
