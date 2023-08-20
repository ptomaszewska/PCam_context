# On the benefit of context in the form of neighbouring tissues on classification performance

The repository contains the source code for the results presented in the paper **On the benefit of context in the form of neighbouring tissues on classification performance** (TBD). In the paper, we investigate the benefit of access to contextual information when doing the classification of histopathological tissues. We also evaluate whether images with different context sizes given to the model may result in different predictions. 

### Setup
We recommend creating a new conda virtual environment:
```
conda create -n PCam_context python=3.8 -y
conda activate PCam_context
pip install -r requirements.txt
```

### Usage

Run the script *run_all_pretrained.sh* which will trigger script *run_single.sh* and consecutively *run_experiment_probs.py* for each analysed model. This way the code can be run in parallel using SLURM queueing system.
In the code, the inference is performed using two types of models (convolutional and transformer-based models). The convolutional models pretrained on histopathological data were taken from TIAToolbox (https://github.com/TissueImageAnalytics/tiatoolbox). The transformer-based models pretrained on Imagenet were later finetuned using the VPT repository (https://github.com/KMnP/vpt) - the details on hyperparameters used are in the paper. 
 
Note that in bash scripts, it is necessary to specify a path to files with data and a path where the output is supposed to be saved.

License This project is under the MIT license. See LICENSE for details.

### Cite
