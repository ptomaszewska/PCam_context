import os
if '/PCam_context' not in os.getcwd():
    os.chdir('./PCam_context/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchmetrics
from sklearn.metrics import silhouette_samples
accuracy = torchmetrics.Accuracy(task='binary')
recall = torchmetrics.Recall(task='binary')
precision = torchmetrics.Precision(task='binary')
f1 = torchmetrics.F1Score(task='binary')
rocauc = torchmetrics.AUROC(task='binary')

HOME_PTH = "./"
NAMEMAP = {'pcamswin':'Swin', 'pcammoco':'MoCo', 'pcamsup':'supViT', 'pcammae':'MAE', 'resnet18-pcam':'ResNet18', 'densenet121-pcam':'DenseNet121'}

def get_pessimistic_swin_lastprob(dataframe_probs):
    swin96_prob = swin224_to_96_prob(dataframe_probs)
    true = torch.Tensor(swin96_prob.loc[:,'label'])
    probs = torch.Tensor(swin96_prob.loc[:,'32'])
    return np.array([met(probs, true).item() for met in [accuracy, f1, precision, recall, rocauc]]), swin96_prob

def swin224_to_96_mets(dataframe_met, dataframe_probs):
    pessimistic_result, transformer_probs = get_pessimistic_swin_lastprob(dataframe_probs)
    dataframe_met.loc[dataframe_met['cut_pixels']==75,'Score'] = pessimistic_result
    dataframe_met.loc[dataframe_met['cut_pixels']==75,'cut_pixels'] = np.array([73.5]*5)
    dataframe_met = dataframe_met[np.logical_and(np.logical_or(dataframe_met['cut_pixels']%7==0, dataframe_met['cut_pixels']==73.5),dataframe_met['cut_pixels']<75)]
    dataframe_met.loc[:,'cut_pixels'] *= 3/7
    dataframe_met.loc[dataframe_met['cut_pixels']==31.5,'cut_pixels'] = np.array([32]*5)
    dataframe_met.loc[:,'cut_pixels'] = dataframe_met.loc[:,'cut_pixels'].astype(int)
    dataframe_met = dataframe_met.reset_index(drop=True)
    return dataframe_met, transformer_probs

def swin224_to_96_prob(dataframe):
    last_results = dataframe['74'].copy()
    index_1 = dataframe.loc[:,'label'] == 1
    last_results[index_1] = dataframe[['74', '75']].min(axis=1).loc[index_1]
    index_0 = dataframe.loc[:,'label'] == 0
    last_results[index_0] = dataframe[['74', '75']].max(axis=1).loc[index_0]
    dataframe['final'] = last_results
    dataframe = dataframe.drop(['74', '75'], axis=1)
    dataframe.columns = ['label', '0', '3', '6', '9', '12', '15', '18', '21', '24', '27', '30','32']
    return dataframe

def standardize_result(dframe):
    for met in ['Accuracy', 'F1_score', 'Precision', 'Recall', 'Roc_auc']:
        dframe.loc[dframe['Metric']==met,'Score'] -= dframe.loc[np.logical_and(dframe['Metric']==met, dframe['cut_pixels']==0),'Score'].iloc[0]
    return dframe

def load_results_probs(modelname, switch_224_to_96=True):
    input_size = 96
    if '-' not in modelname:
        input_size = 224
    pth_model_met = HOME_PTH + f"pretrained_experiments/{modelname}/Experiments_directcall.csv"
    pth_model_probs = HOME_PTH + f"pretrained_experiments/{modelname}/Experiments_directcall_probs.csv"
    exp_result = pd.read_csv(pth_model_met, index_col=0).drop('Data', axis=1)
    exp_result_prob = pd.read_csv(pth_model_probs, index_col=0)
    if input_size == 224 and switch_224_to_96:
        exp_result, exp_result_prob = swin224_to_96_mets(exp_result, exp_result_prob)
    exp_result_standard = standardize_result(exp_result)
    exp_result_standard['Model'] = NAMEMAP[modelname]
    return exp_result_standard, exp_result_prob

def get_pred_type(pred, true):
    if true == 0:
        if pred == 0:
            return 'TN'
        elif pred == 1:
            return 'FP'
    elif true == 1:
        if pred == 1:
            return 'TP'
        elif pred == 0:
            return 'FN'

def melt_probs(dataframe):
    probs_melt_model = pd.melt(dataframe, value_vars=dataframe.drop('label', axis=1).columns, id_vars=['label'], var_name='cut_pixels', value_name='Probability')
    probs_melt_model['cut_pixels'] = probs_melt_model['cut_pixels'].astype(int)
    probs_melt_model['pred'] = (probs_melt_model['Probability'] >= 0.5).astype(int)
    probs_melt_model['pred_type'] = probs_melt_model.apply(lambda x: get_pred_type(x.pred, x.label), axis=1)
    return probs_melt_model

def calculate_silhouette(modelname, save=False):
    probs_model = load_results_probs(modelname, switch_224_to_96=False)[1]
    probs_melt_model = melt_probs(probs_model)
    scores_all = []
    pth_features=HOME_PTH + f'pretrained_experiments/{modelname}/features/'
    cols = sorted([int(x.split('_')[-1][:-3]) for x in os.listdir(pth_features)])
    for col in cols:
        col = int(col)
        feat_col = torch.load(pth_features+f'features_step_{col}.pt')
        pred_col = probs_melt_model[probs_melt_model['cut_pixels']==col]['pred_type']
        score_preds = silhouette_samples(feat_col, labels=pred_col)
        scores_all.append(score_preds)
        print(col, end=' ')
    scores_all=np.concatenate(scores_all)
    if save:
        np.save(HOME_PTH+'pretrained_experiments/'+modelname+'/silhouette.npy', scores_all)
    return


import argparse
parser = argparse.ArgumentParser(description="PCAM pretrained model")
parser.add_argument("name", help="pretrained model name")
args = parser.parse_args()
config = vars(args)
name = config['name']


calculate_silhouette(name, save=True)