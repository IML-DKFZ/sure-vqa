import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


@dataclass
class Metric:
    score_file: str
    score_key: str
    plot_axis: str
    min_score: int
    max_score: int


@dataclass
class Dataset:
    open_closed: str
    raters: list
    train_split: str
    val_split: str


metrics_infos = {
    "Human": Metric("human_metrics_{}_{}.json", "human_score", "Human Score", 1, 5),
    "Mistral": Metric("mistral_metrics.json", "mistral_score", "Mistral Score", 1, 5),
    "BLEU": Metric("open_ended_metrics.json", "bleu_score", "BLEU Score", 0, 1),
    "Exact Match": Metric("open_ended_metrics.json", "exact_match_score", "Exact Match Score", 0, 1),
    "F1": Metric("open_ended_metrics.json", "f1_score", "F1 Score", 0, 1),
    "Precision": Metric("open_ended_metrics.json", "precision", "Precision", 0, 1),
    "Recall": Metric("open_ended_metrics.json", "recall", "Recall", 0, 1),
}


def get_human_automated_metrics_df(dataset:Dataset, metrics_dict:Dict, score_base_path:Path):
    all_human_scores = None
    for rater in dataset.raters:
        human_score_files = [score_base_path / f"human_metrics_{dataset.open_closed}_{rater}.json"]
        human_scores = []
        for human_score_file in human_score_files:
            try:
                with open(human_score_file, "r") as f:
                    human_scores.extend(json.load(f))
            except:
                print(f"Could not open {human_score_file}")
        human_scores_df = pd.DataFrame(human_scores).set_index("qid")

        for metric_name, metric_info in metrics_dict.items():
            if metric_name == "Human":
                continue
            metric_score_file = score_base_path / metric_info.score_file
            with open(metric_score_file, "r") as f:
                metric_scores = json.load(f)
            metric_scores_df = pd.DataFrame(metric_scores).set_index("qid")
            metric_score_key = metric_info.score_key
            metric_scores_df[metric_score_key] = pd.to_numeric(metric_scores_df[metric_score_key], errors='coerce')

            human_scores_df[metric_score_key] = None
            for row in human_scores_df.iterrows():
                qid = row[0]
                metric_score = metric_scores_df.loc[qid][metric_score_key]
                human_scores_df.at[qid, metric_score_key] = metric_score
        if all_human_scores is None:
            all_human_scores = human_scores_df
        else:
            all_human_scores = pd.concat([all_human_scores, human_scores_df])
    return all_human_scores


def get_human_human_metrics_df(dataset:Dataset, score_base_path:Path):
    human_score_files_1 = [score_base_path / f"human_metrics_{dataset.open_closed}_{dataset.raters[0]}.json"]
    human_score_files_2 = [score_base_path / f"human_metrics_{dataset.open_closed}_{dataset.raters[1]}.json"]
    human_scores_1 = []
    human_scores_2 = []
    for human_score_file in human_score_files_1:
        try:
            with open(human_score_file, "r") as f:
                human_scores_1.extend(json.load(f))
        except:
            print(f"Could not open {human_score_file}")
    for human_score_file in human_score_files_2:
        try:
            with open(human_score_file, "r") as f:
                human_scores_2.extend(json.load(f))
        except:
            print(f"Could not open {human_score_file}")
    human_scores_1_df = pd.DataFrame(human_scores_1).set_index("qid")
    human_scores_2_df = pd.DataFrame(human_scores_2).set_index("qid")
    human_scores_1_df["human_score_2"] = None
    for row in human_scores_1_df.iterrows():
        qid = row[0]
        human_score_2 = human_scores_2_df.loc[qid]["human_score"]

        human_scores_1_df.at[qid, "human_score_2"] = human_score_2
    return human_scores_1_df


def get_human_human_correlation(human_human_metrics_df):
    na_mask = ~human_human_metrics_df["human_score"].isna() & \
              ~human_human_metrics_df["human_score_2"].isna()
    kendall, p_val_kendall = kendalltau(
        human_human_metrics_df[na_mask]["human_score"],
        human_human_metrics_df[na_mask]["human_score_2"])
    spearman, p_val_spearman = spearmanr(
        human_human_metrics_df[na_mask]["human_score"],
        human_human_metrics_df[na_mask]["human_score_2"])
    return {
        "kendall": kendall,
        "kendall_p": p_val_kendall,
        "spearman": spearman,
        "spearman_p": p_val_spearman
    }


def correlation_matrix(dataset_dict: Dict, metrics_dict: Dict, exp_base_dir: Path, finetune_method:str, hyperparam_str:str):
    dataset_correlations = {}
    for dataset_name, dataset in dataset_dict.items():
        score_base_path = exp_base_dir / f"{dataset_name}/{finetune_method}/llava-{dataset_name}_train_{dataset.train_split}-finetune_{finetune_method}_{hyperparam_str}/eval/{dataset_name}_val_{dataset.val_split}"
        kendall_df = pd.DataFrame(columns=list(metrics_dict.keys()), index=list(metrics_dict.keys()))
        kendall_p_df = kendall_df.copy()
        spearman_df = kendall_df.copy()
        spearman_p_df = kendall_df.copy()
        human_automated_metrics_df = get_human_automated_metrics_df(dataset, metrics_dict, score_base_path)
        if len(dataset.raters) == 2:
            human_human_metrics_df = get_human_human_metrics_df(dataset, score_base_path)
        else:
            print(f"Not two human raters for {dataset_name}. Only implemented for two raters.")
        for metric_name_row, metric_info_row in metrics_dict.items():
            for metric_name_column, metric_info_column in metrics_dict.items():
                if metric_name_row == "Human" and metric_name_column == "Human":
                    continue
                na_mask = ~human_automated_metrics_df[metric_info_row.score_key].isna() & \
                       ~human_automated_metrics_df[metric_info_column.score_key].isna()
                kendall, p_val_kendall = kendalltau(
                    human_automated_metrics_df[na_mask][metric_info_row.score_key],
                    human_automated_metrics_df[na_mask][metric_info_column.score_key])
                kendall_df.at[metric_name_row, metric_name_column] = kendall
                kendall_p_df.at[metric_name_row, metric_name_column] = p_val_kendall
                spearman, p_val_spearman = spearmanr(
                    human_automated_metrics_df[na_mask][metric_info_row.score_key],
                    human_automated_metrics_df[na_mask][metric_info_column.score_key])
                spearman_df.at[metric_name_row, metric_name_column] = spearman
                spearman_p_df.at[metric_name_row, metric_name_column] = p_val_spearman
        if len(dataset.raters) == 2:
            human_human_correlation = get_human_human_correlation(human_human_metrics_df)
            kendall_df.at["Human", "Human"] = human_human_correlation["kendall"]
            kendall_p_df.at["Human", "Human"] = human_human_correlation["kendall_p"]
            spearman_df.at["Human", "Human"] = human_human_correlation["spearman"]
            spearman_p_df.at["Human", "Human"] = human_human_correlation["spearman_p"]
        dataset_correlations[dataset_name] = {
            "human_automated_metrics_df": human_automated_metrics_df,
            "human_human_metrics_df": human_human_metrics_df if len(dataset.raters) == 2 else None,
            "kendall_df": kendall_df,
            "kendall_p_df": kendall_p_df,
            "spearman_df": spearman_df,
            "spearman_p_df": spearman_p_df,
        }
    return dataset_correlations


def format_p_value(val):
    if val < 0.01:
        return f"{val:.2e}"
    else:
        return f"{val:.2f}"


def single_correlation_heatmap(correlation_df, correlation_p_df, title:str, save_path:Path, min_value=None, max_value=None, plot_p_values=True):
    correlation_df = correlation_df.apply(pd.to_numeric)
    correlation_p_df = correlation_p_df.apply(pd.to_numeric)

    if plot_p_values:
        correlation_df_formatted = correlation_df.round(2).astype(str) + "\n (" + correlation_p_df.map(format_p_value).astype(str) + ")"
        fontsize = "xx-small"
    else:
        correlation_df_formatted = correlation_df.round(2).astype(str)
        fontsize = "large"
    sns.heatmap(correlation_df, annot=correlation_df_formatted, fmt="", annot_kws={"fontsize": fontsize}, square=True, vmin=min_value, vmax=max_value, cbar_kws={"orientation": "horizontal"})
    plt.title(title)
    plt.xticks(rotation=20)
    plt.yticks(rotation=45)
    plt.savefig(save_path)
    plt.close()


def heatmap_plots(dataset_correlations:Dict, plt_save_dir:Path):
    for dataset_name, dataset_correlation in dataset_correlations.items():
        min_value = min(dataset_correlation["kendall_df"].min().min(), dataset_correlation["spearman_df"].min().min())
        single_correlation_heatmap(dataset_correlation["kendall_df"].iloc[:2, :], dataset_correlation["kendall_p_df"].iloc[:2, :],
                                    f"Kendall Correlation", plt_save_dir / f"{dataset_name}_kendall_correlation.png", min_value)
        single_correlation_heatmap(dataset_correlation["spearman_df"].iloc[:2, :], dataset_correlation["spearman_p_df"].iloc[:2, :],
                                    f"Spearman Correlation", plt_save_dir / f"{dataset_name}_spearman_correlation.png", min_value)


def heatmap_plots_small(dataset_correlations:Dict, plt_save_dir:Path):
    kendall_human_dfs = [dataset_correlation["kendall_df"].loc["Human", :] for dataset_correlation in dataset_correlations.values()]
    kendall_human_dfs = pd.concat(kendall_human_dfs)
    kendall_human_dfs_min = kendall_human_dfs.min()
    kendall_human_dfs_max = kendall_human_dfs.max()
    for dataset_name, dataset_correlation in dataset_correlations.items():
        single_correlation_heatmap(dataset_correlation["kendall_df"].iloc[:1, :], dataset_correlation["kendall_p_df"].iloc[:1, :],
                                    f"Kendall Correlation",
                                   plt_save_dir / f"{dataset_name}_kendall_correlation_small.png",
                                   kendall_human_dfs_min, kendall_human_dfs_max, plot_p_values=False)


def single_scatter_plot(correlation_df, metric_name_1, metric_name_2, title:str, save_path:Path):
    if metric_name_1 == "Human" and metric_name_2 == "Human":
        metric_key_1 = "human_score"
        metric_key_2 = "human_score_2"
    else:
        metric_key_1 = metrics_infos[metric_name_1].score_key
        metric_key_2 = metrics_infos[metric_name_2].score_key
    metrics_1_min = int(metrics_infos[metric_name_1].min_score)
    metrics_1_max = int(metrics_infos[metric_name_1].max_score)
    metrics_2_min = int(metrics_infos[metric_name_2].min_score)
    metrics_2_max = int(metrics_infos[metric_name_2].max_score)

    counts = correlation_df.groupby([metric_key_1, metric_key_2]).size().reset_index(name='counts')

    ax_limits = ((metrics_1_min, metrics_2_min), (metrics_1_max, metrics_2_max))
    plt.axline(ax_limits[0], ax_limits[1], color='black', linestyle='--', alpha=0.5, zorder=0)
    plt.scatter(counts[metric_key_1], counts[metric_key_2], s=counts['counts'] * 100, alpha=1, edgecolors='w',
                linewidth=0.5, zorder=1)

    plt.xlim(metrics_1_min-0.5, metrics_1_max+0.5)
    plt.ylim(metrics_2_min-0.5, metrics_2_max+0.5)
    unit_step = int(10*metrics_1_max/5)
    plt.xticks([x / 10.0 for x in range(metrics_1_min*10, (metrics_1_max*10)+unit_step, unit_step)])
    unit_step = int(10*metrics_2_max/5)
    plt.yticks([x / 10.0 for x in range(metrics_2_min * 10, (metrics_2_max * 10) + unit_step, unit_step)])
    plt.xlabel(metric_name_1)
    plt.ylabel(metric_name_2)
    plt.title(title)
    lgnd = plt.legend(["perfect correlation", "measured correlation"], loc='upper left')
    lgnd.legend_handles[1].set_sizes([30])
    plt.savefig(save_path)
    plt.close()


def scatter_plots(dataset_correlations:Dict, plt_save_dir:Path, metrics_correlations: List[Tuple[str, str]]):
    for dataset_name, dataset_correlation in dataset_correlations.items():
        for metrics_correlation in metrics_correlations:
            if metrics_correlation[0] == "Human" and metrics_correlation[1] == "Human":
                if dataset_correlation["human_human_metrics_df"] is None:
                    continue
                correlation_df = dataset_correlation["human_human_metrics_df"]
            else:
                correlation_df = dataset_correlation["human_automated_metrics_df"]
            single_scatter_plot(correlation_df, metrics_correlation[0], metrics_correlation[1],
                                f"{dataset_name} {metrics_correlation[0]} vs {metrics_correlation[1]}",
                                plt_save_dir / f"{dataset_name}_{metrics_correlation[0]}_{metrics_correlation[1]}_scatter.png")


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_root_dir",
        "-e",
        type=str,
        help="Path to the root directory of the experiments",
        default=os.getenv("EXPERIMENT_ROOT_DIR")
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to the folder where the plots will be saved",
        required=True,
    )
    parser.add_argument(
        "--raters",
        "-r",
        type=str,
        help="Raters to include in the analysis. Exactly two raters should be specified.",
        nargs=2,
        required=True
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        help="Finetune method with the human ratings",
        default="ia3"
    )
    parser.add_argument(
        "--hyperparam_str",
        type=str,
        help="Hyperparameter string of the finetuning method with the human ratings",
        default="lr3e-2_seed123"
    )
    args = parser.parse_args()
    return args

def main(args):
    save_dir = Path(args.save_path)
    datasets = {
        "SLAKE": Dataset("open", args.raters, "all", "all"),
        "OVQA": Dataset("open", args.raters, "all", "all"),
        "MIMIC": Dataset("open", args.raters, "sample", "sample"),
    }
    dataset_correlations = correlation_matrix(dataset_dict=datasets, metrics_dict=metrics_infos,
                                              exp_base_dir=Path(args.experiment_root_dir),
                                              finetune_method=args.finetune_method, hyperparam_str=args.hyperparam_str)
    heatmap_plots(dataset_correlations, plt_save_dir=save_dir)
    heatmap_plots_small(dataset_correlations, plt_save_dir=save_dir)
    scatter_plots(dataset_correlations, plt_save_dir=save_dir,
                  metrics_correlations=[("Human", "Mistral"), ("Human", "Human"), ("Human", "BLEU"),
                                        ("Human", "Exact Match"), ("Human", "F1"), ("Human", "Precision"),
                                        ("Human", "Recall")])

if __name__ == "__main__":
    cli_args = main_cli()
    main(args=cli_args)
