# On the benefit of context in the form of neighbouring tissues on classification performance

The repository contains the source code for the results presented in the paper **Does context matter in digital pathology?**. In the paper, we investigate the benefit of access to contextual information when doing the classification of histopathological tissues. We also evaluate whether images with different context sizes given to the model may result in different predictions. 

### Setup
We recommend creating new conda virtual environments for the project. Unfortunately, it is not possible to run all the experiments with one environment due to conflicts in packages. The procedure described in ```env_setup.sh``` can be used to create environment suitable to run all experiments except the ones with MAE model that requires the environment that can be set up using commands specified in ```env_setup_MAE.sh```.

### Usage

Run the script *run_all_pretrained.sh* which will trigger script *run_single.sh* and consecutively *run_experiment_probs.py* for each analysed model. This way the code can be run in parallel using SLURM queueing system.
In the code, the inference is performed using two types of models (convolutional and transformer-based models). The convolutional models pretrained on histopathological data were taken from TIAToolbox (https://github.com/TissueImageAnalytics/tiatoolbox). The transformer-based models pretrained on Imagenet were later finetuned using the VPT repository (https://github.com/KMnP/vpt) - the details on hyperparameters used are in the paper. 
 
Note that in bash scripts, it is necessary to specify a path to files with data and a path where the output is supposed to be saved.

License This project is under the MIT license. See LICENSE for details.
