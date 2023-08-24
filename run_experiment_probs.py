import os
if 'PCam_context' not in os.getcwd():
    os.chdir('PCam_context')
import torch
from experiments import linear_context_experiment
import gc
from models import get_transformer_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse
parser = argparse.ArgumentParser(description="PCAM pretrained model")
parser.add_argument("name", help="pretrained model name")
args = parser.parse_args()
config = vars(args)
name = config['name']

if '-' in name:
    from tiatoolbox.models.engine.patch_predictor import get_pretrained_model

    input_size = 96
    steps = 32
    step_size = 1
    additional_steps = []
    model, config = get_pretrained_model(name)
    model.to(device)
    model.eval()
    def predict_and_features(x):
        x = model.feat_extract(x)
        x = model.pool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = model.classifier(x)
        return torch.softmax(x, -1), feature
else:
    input_size = 224
    steps = 75
    step_size = 7
    additional_steps = [74,75]

    model = get_transformer_model(name)
    model.to(device)
    model.eval()
    def predict_and_features(x):
        if name == 'pcamsup':
            features = model.transformer(x)[0][:, 0]
            preds = model.head(features)
        else:
            features = model.forward_features(x)
            if name == 'pcamswin':
                preds = model.head(features)
            elif name == 'pcammae':
                preds = model.forward(x)
            else:
                preds = model.forward_head(features)
                features = features[:,0]
        return preds, features
        
csv_name = 'Experiments_directcall.csv'
linear_context_experiment(predict_and_features, save=True, pretrained=name, csv_name=csv_name, batch_size=32, input_size=input_size, steps=steps, step_size=step_size, additional_steps=additional_steps)
