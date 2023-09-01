import os
if "/PCam_context" not in os.getcwd():
    os.chdir("../PCam_context/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchmetrics
import warnings

warnings.filterwarnings("ignore")


HOME_PATH = "./"
PATH_save_plots = "./plots/"
# Model names in experiments
models = [
    "pcamswin",
    "pcammoco",
    "pcamsup",
    "pcammae",
    "resnet18-pcam",
    "densenet121-pcam",
]
# Map model names for visualizations
NAMEMAP = {
    "pcamswin": "Swin",
    "pcammoco": "MoCo",
    "pcamsup": "supViT",
    "pcammae": "MAE",
    "resnet18-pcam": "ResNet18",
    "densenet121-pcam": "DenseNet121",
}


accuracy = torchmetrics.Accuracy(task="binary")
recall = torchmetrics.Recall(task="binary")
precision = torchmetrics.Precision(task="binary")
f1 = torchmetrics.F1Score(task="binary")
rocauc = torchmetrics.AUROC(task="binary")


def get_pessimistic_swin_lastprob(dataframe_probs):
    """Transform and Calculate the pessimistic probabilities for the last transformer step.

    Args:
        dataframe_probs (pd.DataFrame): dataframe of the experiments probabilities results.

    Returns:
        tuple of (np.array, pd.DataFrame): tuple consisting of an array of the pessimistic metrics and the probabilities frame.
    """
    swin96_prob = swin224_to_96_prob(dataframe_probs)
    true = torch.Tensor(swin96_prob.loc[:, "label"])
    probs = torch.Tensor(swin96_prob.loc[:, "32"])
    return (
        np.array(
            [
                met(probs, true).item()
                for met in [accuracy, f1, precision, recall, rocauc]
            ]
        ),
        swin96_prob,
    )


def swin224_to_96_mets(dataframe_met, dataframe_probs):
    """Transforms the transformer results with size of 224 into convolution compatible format for comparison. Every 7th pixel on 224 image size corresponds to every 3rd pixel on 96 original image size. Last step on convolutions, 32, is between 74th and 75th pixel on transformers, therefore the furthest probability from the correct label is used.

    Args:
        dataframe_met (pd.DataFrame): Metrics dataframe for one of the 224 models.
        dataframe_probs (pd.DataFrame): Probabilities dataframe for one of the 224 models.

    Returns:
        tuple of (pd.DataFrame, pd.DataFrame): A tuple consisting of transformed results, first consists of the metrics and the second of probabilities.
    """
    pessimistic_result, transformer_probs = get_pessimistic_swin_lastprob(
        dataframe_probs
    )
    dataframe_met.loc[dataframe_met["cut_pixels"] == 75, "Score"] = pessimistic_result
    dataframe_met.loc[dataframe_met["cut_pixels"] == 75, "cut_pixels"] = np.array(
        [73.5] * 5
    )
    dataframe_met = dataframe_met[
        np.logical_and(
            np.logical_or(
                dataframe_met["cut_pixels"] % 7 == 0,
                dataframe_met["cut_pixels"] == 73.5,
            ),
            dataframe_met["cut_pixels"] < 75,
        )
    ]
    dataframe_met.loc[:, "cut_pixels"] *= 3 / 7
    dataframe_met.loc[dataframe_met["cut_pixels"] == 31.5, "cut_pixels"] = np.array(
        [32] * 5
    )
    dataframe_met.loc[:, "cut_pixels"] = dataframe_met.loc[:, "cut_pixels"].astype(int)
    dataframe_met = dataframe_met.reset_index(drop=True)
    return dataframe_met, transformer_probs


def swin224_to_96_pred(dataframe):
    """Transform the transformer 224 format predictions to convolutional format of 96.

    Args:
        dataframe (pd.DataFrame): dataframe of predictions.

    Returns:
        pd.DataFrame: dataframe of transformed predictions.
    """
    last_results = dataframe["label"].copy()
    error_74 = dataframe.loc[:, "74"] != dataframe.loc[:, "label"]
    last_results[error_74] = dataframe.loc[error_74, "74"]
    error_75 = dataframe.loc[:, "75"] != dataframe.loc[:, "label"]
    last_results[error_75] = dataframe.loc[error_75, "75"]
    dataframe["final"] = last_results
    dataframe = dataframe.drop(["74", "75"], axis=1)
    dataframe.columns = [
        "label",
        "0",
        "3",
        "6",
        "9",
        "12",
        "15",
        "18",
        "21",
        "24",
        "27",
        "30",
        "32",
    ]
    return dataframe


def swin224_to_96_prob(dataframe):
    """_summary_

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    last_results = dataframe["74"].copy()
    index_1 = dataframe.loc[:, "label"] == 1
    last_results[index_1] = dataframe[["74", "75"]].min(axis=1).loc[index_1]
    index_0 = dataframe.loc[:, "label"] == 0
    last_results[index_0] = dataframe[["74", "75"]].max(axis=1).loc[index_0]
    dataframe["final"] = last_results
    dataframe = dataframe.drop(["74", "75"], axis=1)
    dataframe.columns = [
        "label",
        "0",
        "3",
        "6",
        "9",
        "12",
        "15",
        "18",
        "21",
        "24",
        "27",
        "30",
        "32",
    ]
    return dataframe


def standardize_result(dframe):
    """Standardize the results to the metric results on the experiments with full images.

    Args:
        dframe (pd.DataFrame): dataframe with the metrics results.

    Returns:
        pd.DataFrame: standardized results.
    """
    for met in ["Accuracy", "F1_score", "Precision", "Recall", "AUC"]:
        dframe.loc[dframe["Metric"] == met, "Score"] -= dframe.loc[
            np.logical_and(dframe["Metric"] == met, dframe["cut_pixels"] == 0), "Score"
        ].iloc[0]
    return dframe


def add_rocauc_224(model):
    """Function for adding the roc auc metric after experiments run, using probabilities.

    Args:
        model (str): name of the model.

    Returns:
        tuple of (pd.DataFrame, str): tuple with the new frame of metric results and the path to metrics.
    """
    model_pth_probs = (
        HOME_PATH + f"pretrained_experiments/{model}/Experiments_directcall_probs.csv"
    )
    model_pth_mets = (
        HOME_PATH
        + f"pretrained_experiments/{model}/Experiments_directcall_uniquedata.csv"
    )
    probs = pd.read_csv(model_pth_probs, index_col=0).drop_duplicates()
    metrics = pd.read_csv(model_pth_mets, index_col=0)
    unique_pixels = metrics["cut_pixels"].unique()
    roc_dict = {
        "Score": [],
        "Metric": ["AUC"] * len(unique_pixels),
        "cut_pixels": unique_pixels,
        "Data": ["test"] * len(unique_pixels),
    }
    for pixel in unique_pixels:
        roc_dict["Score"].append(
            rocauc(torch.Tensor(probs[str(pixel)]), torch.Tensor(probs["label"])).item()
        )
    roc_dict = pd.DataFrame(roc_dict)
    model_results = (
        pd.concat([metrics, roc_dict])
        .sort_values(["cut_pixels", "Metric"])
        .reset_index(drop=True)
    )
    return model_results, model_pth_mets


def load_results_probs(modelname):
    """Load the saved experiments results into proper plotting formats.

    Args:
        modelname (str): name of the model.

    Returns:
        tuple of types (pd.DataFrame, pd.DataFrame): tuple of pandas Dataframes, first one consisting of the experiment metrics and second one containing prediction probabilities.
    """
    input_size = 96
    if "-" not in modelname:
        input_size = 224
    pth_model_met = (
        HOME_PATH
        + f"pretrained_experiments/{modelname}/Experiments_directcall_uniquedata.csv"
    )
    pth_model_probs = (
        HOME_PATH
        + f"pretrained_experiments/{modelname}/Experiments_directcall_probs.csv"
    )
    exp_result = pd.read_csv(pth_model_met, index_col=0).drop("Data", axis=1)
    exp_result = exp_result.replace("Roc_auc", "AUC")
    exp_result_prob = pd.read_csv(pth_model_probs, index_col=0).drop_duplicates()
    if input_size == 224:
        exp_result, exp_result_prob = swin224_to_96_mets(exp_result, exp_result_prob)
    exp_result_standard = standardize_result(exp_result)
    exp_result_standard["Model"] = NAMEMAP[modelname]
    return exp_result_standard, exp_result_prob


def plot_single_standard(modelname):
    """Plot a single models metrics.

    Args:
        modelname (str): Name of the model to plot.
    """
    exp_result_standard, _ = load_results_probs(modelname)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(exp_result_standard, y="Score", x="cut_pixels", hue="Metric", ax=ax)
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    plt.title("Performance - " + NAMEMAP[modelname])
    plt.xlabel("Context size")
    ax.set_xticks(ticks=np.arange(0, 33, 4), labels=np.arange(32, -1, -4))
    # plt.savefig(PATH_save_plots+NAMEMAP[modelname]+'.png', dpi=600, bbox_inches='tight')


def plot_multiple(models, sharey=False):
    """Plot multiple models metrics on subplots.

    Args:
        models (_type_): Name of the models to plot.
        sharey (bool, optional): Whether to share y axis. Defaults to False.
    """
    joined_results = pd.concat([load_results_probs(model)[0] for model in models])
    fig, ax = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 7), sharey=sharey, sharex=True
    )
    for met, axis in zip(
        ["Accuracy", "Recall", "Precision", "AUC"],
        [ax[0][0], ax[0][1], ax[1][0], ax[1][1]],
    ):
        print(
            joined_results[
                (joined_results.Metric == met) & (joined_results.cut_pixels == 32)
            ]
        )
        sns.lineplot(
            joined_results[joined_results["Metric"] == met],
            y="Score",
            x="cut_pixels",
            hue="Model",
            ax=axis,
        )
        for transf_name in ["Swin", "MoCo", "supViT", "MAE"]:
            sns.scatterplot(
                joined_results[
                    np.logical_and(
                        joined_results["Model"] == transf_name,
                        joined_results["Metric"] == met,
                    )
                ],
                y="Score",
                x="cut_pixels",
                ax=axis,
            )
        axis.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axis.title.set_text(met)
        axis.set_xlabel("Context size")
        axis.set_ylabel("Difference in performance")
        axis.set_xticks(ticks=np.arange(0, 33, 4), labels=np.arange(32, -1, -4))

    fig.tight_layout()
    # plt.savefig(f"{PATH_save_plots}all_models_performance_{sharey}.png", dpi=600, bbox_inches='tight')


def get_pred_type(pred, true):
    """Returns the prediction type for the prediction and true label. Values in ["TN", "FP", "TP", "FN"].

    Args:
        pred (int): prediction class
        true (int): true label class

    Returns:
        str: type of classifications prediction
    """
    if true == 0:
        if pred == 0:
            return "TN"
        elif pred == 1:
            return "FP"
    elif true == 1:
        if pred == 1:
            return "TP"
        elif pred == 0:
            return "FN"


def melt_probs(dataframe):
    """Melt the dataframe of probabilities and add the prediction type.

    Args:
        dataframe (pd.DataFrame): probabilities dataframe in the wide format.

    Returns:
        pd.DataFrame: probabilities dataframe in the long format.
    """
    probs_melt_model = pd.melt(
        dataframe,
        value_vars=dataframe.drop("label", axis=1).columns,
        id_vars=["label"],
        var_name="cut_pixels",
        value_name="Probability",
    )
    probs_melt_model["cut_pixels"] = probs_melt_model["cut_pixels"].astype(int)
    probs_melt_model["pred"] = (probs_melt_model["Probability"] >= 0.5).astype(int)
    probs_melt_model["pred_type"] = probs_melt_model.apply(
        lambda x: get_pred_type(x.pred, x.label), axis=1
    )
    return probs_melt_model


def walking_change(df_melted):
    """Calculate the change from the previous step in the probabilities.

    Args:
        df_melted (pd.DataFrame): Dataframe of probabilities in the long format

    Returns:
        pd.DataFrame: Dataframe of the changes in probabilities, with the change of prediction type.
    """
    probs_melt_next = df_melted[df_melted["cut_pixels"] != 32].join(
        df_melted[df_melted["cut_pixels"] != 0].reset_index(drop=True), rsuffix="_next"
    )
    probs_melt_next["predtype_change"] = (
        probs_melt_next["pred_type"] + "->" + probs_melt_next["pred_type_next"]
    )
    probs_melt_next["prob_change"] = (
        probs_melt_next["Probability"] - probs_melt_next["Probability_next"]
    )
    return probs_melt_next

    fig, ax = plt.subplots(figsize=(5, 5), nrows=2, ncols=2, sharex=True, sharey="row")
    for i, model in enumerate(modelname):
        probs_model = load_results_probs(model)[1]
        probs_melt_model = melt_probs(probs_model)
        counts_groups_model = (
            probs_melt_model.groupby(["pred_type", "cut_pixels"])
            .size()
            .reset_index(name="counts")
        )

        probs_melt_model.sort_values(by=["pred_type", "cut_pixels"], inplace=True)
        counts_groups_model.sort_values(by=["pred_type", "cut_pixels"], inplace=True)

        if i == 0:
            sns.lineplot(
                probs_melt_model,
                x="cut_pixels",
                y="Probability",
                hue="pred_type",
                ax=ax[0][i],
            )
            handles, labels = ax[0][0].get_legend_handles_labels()
            ax[i][0].get_legend().remove()
        else:
            sns.lineplot(
                probs_melt_model,
                x="cut_pixels",
                y="Probability",
                hue="pred_type",
                ax=ax[0][i],
                legend=False,
            )
        ax[0][i].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        sns.lineplot(
            counts_groups_model,
            x="cut_pixels",
            y="counts",
            hue="pred_type",
            ax=ax[1][i],
            legend=False,
        )
        ax[0][i].title.set_text(NAMEMAP[model])
        ax[1][i].set_xlabel("Context size")
        ax[1][0].set_ylabel("Count")
        ax[i][0].set_xticks(ticks=np.arange(0, 33, 8), labels=np.arange(32, -1, -8))

    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)

    # plt.savefig(PATH_save_plots+'probs_4categories.png', dpi=600, bbox_inches='tight')


def plot_walkingchange(modelname, hue="label"):
    """Plot the walking change of probabilities for a single model.

    Args:
        modelname (str): Name of the model to plot.
        hue (str, optional): The column to group by for the plotting. Defaults to "label".

    Returns:
        tuple of (pd.DataFrame, pd.Dataframe): tuple of the probabilities dataframe and the melted change dataframe.
    """
    if hue not in [
        "label",
        "pred",
        "pred_type",
        "label_next",
        "pred_next",
        "pred_type_next",
        "predtype_change",
    ]:
        return
    probs_model = load_results_probs(modelname)[1]
    probs_melt_model = melt_probs(probs_model)
    walking_change_model_melt = walking_change(probs_melt_model)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(
        walking_change_model_melt, x="cut_pixels", y="prob_change", hue=hue, ax=ax
    )
    ax.title.set_text(NAMEMAP[modelname])
    plt.show()
    return probs_model, walking_change_model_melt


def plot_predtypes(modelname, with_markers=False):
    """Plot the prediction types for a single model.

    Args:
        modelname (str): Name of the model to plot.
        with_markers (bool, optional): Whether to add markers for the data points. Defaults to False.
    """
    if with_markers == True:
        figsize = (15, 15)
        step = 2
    else:
        figsize = (5, 5)
        step = 8
    fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=2, sharex=True, sharey="row")
    for i, model in enumerate(modelname):
        probs_model = load_results_probs(model)[1]
        probs_melt_model = melt_probs(probs_model)
        counts_groups_model = (
            probs_melt_model.groupby(["pred_type", "cut_pixels"])
            .size()
            .reset_index(name="counts")
        )

        probs_melt_model.sort_values(by=["pred_type", "cut_pixels"], inplace=True)
        counts_groups_model.sort_values(by=["pred_type", "cut_pixels"], inplace=True)

        if model in ["resnet18-pcam", "densenet121-pcam"]:
            marker = None
        elif with_markers == True:
            marker = "o"

        if i == 0:
            sns.lineplot(
                probs_melt_model,
                x="cut_pixels",
                y="Probability",
                hue="pred_type",
                ax=ax[0][i],
            )
            handles, labels = ax[0][0].get_legend_handles_labels()
            ax[i][0].get_legend().remove()
        else:
            sns.lineplot(
                probs_melt_model,
                x="cut_pixels",
                y="Probability",
                hue="pred_type",
                ax=ax[0][i],
                legend=False,
            )
        ax[0][i].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        sns.lineplot(
            counts_groups_model,
            x="cut_pixels",
            y="counts",
            hue="pred_type",
            ax=ax[1][i],
            legend=False,
            marker=marker,
        )
        ax[0][i].title.set_text(NAMEMAP[model])
        ax[1][i].set_xlabel("Context size")
        ax[i][0].set_xticks(
            ticks=np.arange(0, 33, step), labels=np.arange(32, -1, -step)
        )

    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)

    # plt.savefig(PATH_save_plots+'probs_4categories.png', dpi=600, bbox_inches='tight')
    plt.show()


def changing_probs_sample_images(probs_model_all, changes_all, model_index):
    """Plot the changing probabilities for sample most often changing images.

    Args:
        probs_model_all (pd.DataFrame): probabilities for all models.
        changes_all (pd.DataFrame): changes dataframe in the wide format.
        model_index (pd.index): index for the specific model.
    """
    probs_model = probs_model_all[model_index]
    changes_sum = changes_all[model_index].sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5), nrows=2, ncols=1, sharex=True)
    sns.lineplot(
        probs_model[list(changes_sum > 16)].drop("label", axis=1).drop_duplicates().T,
        legend=False,
        ax=ax[0],
    )
    ax[0].axhline(y=0.5, color="black", linestyle="--", alpha=0.5, lw=2)
    ax[0].set_xticks(ticks=np.arange(0, 33, 8), labels=np.arange(32, -1, -8))
    ax[0].set_title("Images whose prediction changes more than 16 times")

    changing_once = probs_model[list(changes_sum == 1)].drop("label", axis=1)
    sns.lineplot(
        changing_once[changing_once.index.isin([398, 5127, 24840, 1446, 21005])].T,
        ax=ax[1],
        legend=False,
    )
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, lw=2)
    ax[1].set_title("Images whose prediction changes only once")
    ax[1].set_xlabel("Context size")

    modelname = models[model_index]

    # plt.savefig(f"{PATH_save_plots}probabilities_changing_{modelname}.png", dpi=600, bbox_inches='tight')
    plt.show()


def plot_n_changes_context(changes_all):
    """Plot the number of changes for all models.

    Args:
        changes_all (pd.DataFrame): changes dataframe in the wide format.

    Returns:
        pd.DataFrame: melted sum of changes in the long format.
    """
    changes_sum_melt_all = []
    for j, model in enumerate(models):
        changes = changes_all[j]
        changes_sum = changes.sum(axis=1)
        changes_sum_melt = pd.melt(
            changes[changes_sum == 1].reset_index(), id_vars="index"
        )
        changes_sum_melt_all.append(changes_sum_melt)
        number_images_change = (
            changes_sum_melt[changes_sum_melt["value"] != 0]
            .groupby("cut_pixels")
            .sum()["value"]
        )
        if model in ["resnet18-pcam", "densenet121-pcam"]:
            context_sizes = number_images_change.index[::3]
            context_sizes = context_sizes[:-1] + 3
            number_images_change = np.add.reduceat(
                list(number_images_change[:-2]),
                np.arange(0, len(number_images_change[:-2]), 3),
            ).tolist()
        else:
            context_sizes = number_images_change.index[:-1] + 3
            number_images_change = number_images_change[:-1]

        sns.lineplot(x=context_sizes, y=number_images_change, label=NAMEMAP[model])
    plt.xticks(ticks=np.arange(0, 31, 3), labels=np.arange(30, -1, -3))
    plt.ylabel("Count")
    plt.xlabel("Context size")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    # plt.savefig(f"{PATH_save_plots}number_changing.png", dpi=600, bbox_inches='tight')
    plt.show()
    return changes_sum_melt_all


def plot_histograms(walking_change_all, changes_sum_melt_all):
    """Plot the prediction type change histograms.

    Args:
        walking_change_all (pd.DataFrame): walking change dataframe in the long format.
        changes_sum_melt_all (pd.DataFrame): summed changes dataframe in the long format.
    """
    for j, model in enumerate(models):
        walking_change_model = walking_change_all[j]
        print(model)
        walking_change_model = walking_change_model.merge(
            changes_sum_melt_all[j], on=["cut_pixels", "index"], how="left"
        ).fillna(0)
        data = walking_change_model[walking_change_model["value"] == 1]
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(6, 6), sharex=True, sharey=False
        )

        bins = walking_change_model.cut_pixels_next.unique()

        for predtype, indx in zip(
            ["FN->TP", "TN->FP", "FP->TN", "TP->FN"], [(0, 0), (0, 1), (1, 0), (1, 1)]
        ):
            sns.histplot(
                data[data["predtype_change"] == predtype],
                x="cut_pixels_next",
                ax=ax[indx[0]][indx[1]],
                bins=bins,
            )
            ax[indx[0]][indx[1]].title.set_text(predtype)

        ax[1][0].set_xlabel("Context size")
        ax[1][1].set_xlabel("Context size")
        ax[1][0].set_xticks(ticks=np.arange(0, 33, 8), labels=np.arange(32, -1, -8))
        fig.tight_layout()

        # plt.savefig(f"{PATH_save_plots}histograms_{model}.png", dpi=600, bbox_inches='tight')
        plt.show()


def prediction_changes_per_images(changes_all):
    """Plot the number of prediction changes in logarithimc scale for all models..

    Args:
        changes_all (pd.DataFrame): changes dataframe in the wide format.
    """
    plt.figure(figsize=(6, 4))
    for j, model in enumerate(models):
        changes_sum = changes_all[j].sum(axis=1)
        if model not in ["resnet18-pcam", "densenet121-pcam"]:
            sns.lineplot(
                x=changes_sum.value_counts().index * 3,
                y=np.log(changes_sum.value_counts()),
                label=NAMEMAP[model],
                marker="o",
            )
        else:
            sns.lineplot(
                x=changes_sum.value_counts().index,
                y=np.log(changes_sum.value_counts()),
                label=NAMEMAP[model],
            )
    plt.legend()

    plt.xlabel("Number of times the model changed prediction for the images")
    plt.ylabel("Ln(Count)")
    plt.xticks(range(0, 32, 2))
    # plt.savefig(f"{PATH_save_plots}number_of_all_changes.png", dpi=600, bbox_inches='tight')
    plt.show()
