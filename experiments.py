import os
import gc
import torch
import pandas as pd
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import PCAM
import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
accuracy = torchmetrics.Accuracy(task='binary').to(device)
recall = torchmetrics.Recall(task='binary').to(device)
precision = torchmetrics.Precision(task='binary').to(device)
f1 = torchmetrics.F1Score(task='binary').to(device)
rocauc = torchmetrics.AUROC(task='binary').to(device)

def fill_results(results, probs, true, pixels, data):
    for met, name in zip([accuracy, f1, precision, recall, rocauc], ['Accuracy', 'F1_score', 'Precision', 'Recall', 'Roc_auc']):
        results['Score'].append(met(probs, true).item())
        results['Metric'].append(name)
        results['cut_pixels'].append(pixels)
        results['Data'].append(data)
        if name == 'F1_score':
            print(data, ' F1: ', results['Score'][-1])
    return results

def get_transform(hide_size, color='black', normalize=True, input_size = 96):
    color_d = {'black':0, 'white':1}
    if color in ['black', 'white']:
        pad_transform = transforms.Pad(hide_size, fill=color_d[color])
    transforms_list = []
    if input_size != 96:
        transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    if normalize:
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transforms_list.extend([transforms.CenterCrop(input_size-2*hide_size), pad_transform])
    transform = transforms.Compose(transforms_list)
    return transform

def linear_context_experiment(predictor, steps=32, batch_size=128, num_workers=1, save=True, color='black', pretrained=None, csv_name='Experiments.csv', input_size=96, start_step=0, additional_steps=[],step_size=1):
    pth = ''
    name = ''
    if pretrained is None:
        name = predictor.name
        pth = f'./models/{name}/'
    else:
        name = pretrained
        pth = f'./pretrained_experiments/{name}/'
    features_pth = pth + 'features/'
    os.makedirs(pth, exist_ok=True)
    os.makedirs(features_pth, exist_ok=True)
    pth_csv = pth + csv_name
    results = {'Score':[], 'Metric':[], 'cut_pixels':[], 'Data':[]}
    predicts = {}
    steps_list = list(range(start_step, steps+1, step_size)) + additional_steps
    for i in steps_list:
        transform = get_transform(i, color=color, normalize=False, input_size=input_size)
        for split in ['test']:# ['train', 'test', 'val']:
            currset = PCAM(root='./data', split=split, download=True, transform=transform)
            currloader = torch.utils.data.DataLoader(currset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            curr_probs = torch.Tensor().to(device)
            curr_true = torch.Tensor().to(device)
            curr_features = torch.Tensor()
            for data in currloader:
                images, labels = data[0].to(device), data[1].to(device)
                curr_true = torch.cat((curr_true, labels))
                torch.cuda.empty_cache()
                gc.collect()
                with torch.no_grad():
                    probs, features = predictor(images)
                    if input_size == 224: #models from the toolbox return the probabilities (not logits) unlike transformers
                        probs = torch.softmax(probs, dim=-1)
                    probs_class1 = probs[:,1]
                curr_probs = torch.cat((curr_probs, probs_class1))
                curr_features = torch.cat((curr_features, features.cpu()))
            if save:
                torch.save(curr_features, features_pth + f'features_step_{i}.pt')
            fill_results(results, curr_probs, curr_true, i, split)
            curr_results = pd.DataFrame(results).sort_values(['cut_pixels', 'Metric']).reset_index(drop=True)
            curr_results.to_csv(f'./pretrained_experiments/Temp_res_{name}.csv')
            if split=='test':
                if i == 0:
                    predicts['label'] = curr_true.cpu().numpy()
                predicts[str(i)]= curr_probs.cpu().numpy()
                curr_preds_df = pd.DataFrame(predicts)
                curr_preds_df.to_csv(f'./pretrained_experiments/Temp_probs_{split}_{name}.csv')
        print('done ', i)
    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(predicts)
    if save:
        results_df.to_csv(pth_csv)
        preds_df.to_csv(pth_csv[:-4] + '_probs.csv')
    return results_df, preds_df
    
def plot_experiment_results(results, name, save=True, pretrained=False):
    fig, ax = plt.subplots(nrows=3, figsize=(12,24))
    sns.lineplot(results[results['Data']=='val'], y='Score', x='cut_pixels', hue='Metric', ax=ax[0])
    sns.lineplot(results[results['Data']=='train'], y='Score', x='cut_pixels', hue='Metric', ax=ax[1])
    sns.lineplot(results[results['Data']=='test'], y='Score', x='cut_pixels', hue='Metric', ax=ax[2])
    ax[0].title.set_text('Validation')
    ax[1].title.set_text('Train')
    ax[2].title.set_text('Test')
    for axi in ax:
        axi.set_xlim([-0.1, results['cut_pixels'].max()+1])
        axi.set_ylim([0, 1.01])
    if save:
        if not pretrained:
            pth = f'./models/{name}/Experiment_plots.jpg'
        else:
            pth = f'./pretrained_experiments/{name}/Experiment_plots.jpg'
        plt.savefig(pth)
    plt.show()
    

def experiments_xai(xai_explainer_call, target, predictions, save_pth, steps=32, batch_size=64, start=0, input_size=96):
    def get_target(batch_label, step, batch_num):
        if target == 'label':
            return batch_label
        elif target == 'predictions':
            nparray = predictions.loc[batch_num*batch_size:((batch_num+1)*batch_size-1), str(step)].to_numpy()
            return torch.Tensor(nparray).int().type(torch.int64).to(device)
        else:
            return None
    step_size=1 
    for i in range(start, steps, step_size):
        transform = get_transform(i, color='black', normalize=False, input_size=input_size)
        split='test'
        test_set = PCAM(root='./data', split=split, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        all_heatmaps = torch.Tensor()
        for batch_num, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            batch_target = get_target(labels, i, batch_num)
            torch.cuda.empty_cache()
            gc.collect()
            xai_heatmap = xai_explainer_call(images, batch_target)
            del batch_target, images, labels
            all_heatmaps = torch.cat((all_heatmaps, xai_heatmap.cpu()))
            del xai_heatmap
            torch.cuda.empty_cache()
            gc.collect()
            #print(batch_num, ' done')
            if (batch_num*batch_size)%4096 == 0 and batch_num*batch_size != 0:
                torch.save(all_heatmaps, save_pth+f'heatmap_step_{i}.pt')
        torch.save(all_heatmaps, save_pth+f'heatmap_step_{i}.pt')
    return