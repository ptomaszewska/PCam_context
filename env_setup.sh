#!/bin/bash
apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n PCam_context python=3.8 -y
conda activate PCam_context

conda install -y pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install -y pandas matplotlib seaborn scikit-learn scipy

pip install h5py
pip install torchmetrics
pip install --upgrade torchaudio

pip install ml_collections
pip install timm
pip install torch
conda install -c -y conda-forge pixman
pip install tiatoolbox
