import os
if "PCam_context" not in os.getcwd():
    os.chdir("PCam_context")
import gc
import torch
import pandas as pd
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import PCAM

device = "cuda" if torch.cuda.is_available() else "cpu"
accuracy = torchmetrics.Accuracy(task="binary").to(device)
recall = torchmetrics.Recall(task="binary").to(device)
precision = torchmetrics.Precision(task="binary").to(device)
f1 = torchmetrics.F1Score(task="binary").to(device)
rocauc = torchmetrics.AUROC(task="binary").to(device)


def fill_results(results, probs, true, pixels, data_split):
    """Calculate classification metrics from probabilities tensor and insert into a dictionary, for a specific amount of removed pixels.

    Args:
        results : Dictionary with keys of ["Score", "Metric", "cut_pixels", "Data"].
        probs : Tensor of probabilities to calculate metrics.
        true : Tensor of true labels corresponding to probabilities from predictions.
        pixels (int): Number of cut pixels in the current data.
        data_split (str): Data split, one of ["train", "test", "val"].

    Returns:
        results dictionary appended with the current metric scores.
    """
    for met, name in zip(
        [accuracy, f1, precision, recall, rocauc],
        ["Accuracy", "F1_score", "Precision", "Recall", "Roc_auc"],
    ):
        results["Score"].append(met(probs, true).item())
        results["Metric"].append(name)
        results["cut_pixels"].append(pixels)
        results["Data"].append(data_split)
        if name == "F1_score":
            print(data_split, " F1: ", results["Score"][-1])
    return results


def get_transform(hide_size, input_size=96):
    """Get the dataset transformation required for the current experiment and amount of pixels to remove. If input_size is 96, transformations are ToTensor, center cropping and padding. When input_size is 224, a resize to size of 224 is added first.

    Args:
        hide_size (int): Amount of pixels to remove from the border.
        input_size (int): The size of images required by the used models. Convolutional models in the experiments use size of 96, while transfomers use 224. Defaults to 96.

    Returns:
        Composition of transformations for the current experiment.
    """
    transforms_list = []
    if input_size != 96:
        transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    pad_transform = transforms.Pad(hide_size, fill=0)
    transforms_list.extend(
        [transforms.CenterCrop(input_size - 2 * hide_size), pad_transform]
    )
    transform = transforms.Compose(transforms_list)
    return transform


def linear_context_experiment(
    predictor,
    steps=32,
    batch_size=128,
    num_workers=1,
    save=True,
    model_name=None,
    csv_name="Experiments.csv",
    input_size=96,
    start_step=0,
    additional_steps=[],
    step_size=1,
):
    """Run the experiment for the specific predictor. For each step, data is transformed to the correct shape and using the predictor function both predictions and features are inferred. Metrics of classification and probabilites for each step are saved in pandas.DataFrames. The precise steps used in the experiment are derived as: list(range(start_step, steps + 1, step_size)) + additional_steps.

    Args:
        predictor : A prediction function that returns a tuple of probabilites or scores and features.
        steps (int): Number of steps in the experiment. Default value for convolutional models is 32, while for transformers it should be 75. Defaults to 32.
        batch_size (int): Batch size for inference. Defaults to 128.
        num_workers (int): num_workers for the Data Loader. Defaults to 1.
        save (bool): Whether to save the results of experiments. Defaults to True.
        model_name (str): Name of the model. Used to define and create the path for the results. Defaults to None.
        csv_name (str, optional): Name the csv file for saving the results. Defaults to "Experiments.csv".
        input_size (int, optional): The size of images required by the used models. Convolutional models in the experiments use size of 96, while transfomers use 224. Defaults to 96.
        start_step (int, optional): Starting step for the experiment. Defaults to 0.
        additional_steps (list, optional): List of additional steps to take. Defaults to [].
        step_size (int, optional): Size of steps to take. For convolutional models step_size of 1 was used, while for transformers 7. Defaults to 1.

    Returns:
        tuple of (pandas.DataFrame, pandas.DataFrame): A tuple consisting of two dataframes. First containts the classification metrics results, while the second one contains the labels and probabilities from the experiment.
    """
    pth = f"./pretrained_experiments/{model_name}/"
    features_pth = pth + "features/"
    os.makedirs(pth, exist_ok=True)
    os.makedirs(features_pth, exist_ok=True)
    pth_csv = pth + csv_name
    results = {"Score": [], "Metric": [], "cut_pixels": [], "Data": []}
    predicts = {}
    steps_list = list(range(start_step, steps + 1, step_size)) + additional_steps
    for i in steps_list:
        transform = get_transform(i, input_size=input_size)
        for split in ["test"]:
            currset = PCAM(
                root="./data", split=split, download=True, transform=transform
            )
            currloader = torch.utils.data.DataLoader(
                currset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
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
                    if (
                        input_size == 224
                    ):  # models from the toolbox return the probabilities (not logits) unlike transformers
                        probs = torch.softmax(probs, dim=-1)
                    probs_class1 = probs[:, 1]
                curr_probs = torch.cat((curr_probs, probs_class1))
                curr_features = torch.cat((curr_features, features.cpu()))
            if save:
                torch.save(curr_features, features_pth + f"features_step_{i}.pt")
            fill_results(results, curr_probs, curr_true, i, split)
            curr_results = (
                pd.DataFrame(results)
                .sort_values(["cut_pixels", "Metric"])
                .reset_index(drop=True)
            )
            if save:
                curr_results.to_csv(
                    f"./pretrained_experiments/Temp_res_{model_name}.csv"
                )
            if i == 0:
                predicts["label"] = curr_true.cpu().numpy()
            predicts[str(i)] = curr_probs.cpu().numpy()
            curr_preds_df = pd.DataFrame(predicts)
            curr_preds_df.to_csv(
                f"./pretrained_experiments/Temp_probs_{split}_{model_name}.csv"
            )
        print("done ", i)
    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(predicts)
    if save:
        results_df.to_csv(pth_csv)
        preds_df.to_csv(pth_csv[:-4] + "_probs.csv")
    return results_df, preds_df
