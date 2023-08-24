apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools
conda create -n PCam_context python=3.8 -y
conda activate PCam_context
pip install -r requirements.txt
