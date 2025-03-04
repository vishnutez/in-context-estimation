"""
Based on TwoTaskMixturePlots.ipynb. That notebook performs the experiments in
Section 4.2. This script performs the experiments in Section 4.1, for the
Gaussian mixture distribution.
"""

from collections import OrderedDict
import re
import os
import sys
import copy
import argparse

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler, SparseLinearRegression
from tasks import LinearRegression as LinearRegressionTask


import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor, Ridge
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run_id', help='Run ID (directory name) for the desired run')
parser.add_argument('--normalize_outputs', action='store_true')

args = parser.parse_args()




sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = False

matplotlib.rcParams.update(
    {
        "axes.titlesize": 8,
        "figure.titlesize": 10,  # was 10
        "legend.fontsize": 12,  # was 10
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)
run_dir = "../models/"




SPINE_COLOR = "gray"


def format_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)
    return ax




cuda_device = "cuda:0"

task = 'gaussian_mixture_linear_regression'
run_id = args.run_id
gmm_lr_model, gmm_lr_conf = get_model_from_run(os.path.join(run_dir, task, run_id))
gmm_lr_model.to(cuda_device)





batch_size = 1280  # 1280 #conf.training.batch_size
n_dims = 10
n_points = gmm_lr_conf.training.curriculum.points.end
data_sampler = get_data_sampler(gmm_lr_conf.training.data, n_dims)





seed = 42
torch.manual_seed(seed)

# Note: TwoTaskMixturePlots.ipynb sets normalize_outputs to True here. However,
# it is explicitly set to False in gmm_linear_regression.yaml.
if args.normalize_outputs:
    gmm_lr_conf.training.task_kwargs.update({"normalize_outputs": True})

print(f'normalize_outputs: {gmm_lr_conf.training.task_kwargs.normalize_outputs}')

# Create 2 tasks, for the two Gaussians in the mixture. A hack for doing this is to
# use the same task kwargs but set mixing_ratio to either 0 or 1.
t1_task_kwargs = copy.deepcopy(gmm_lr_conf.training.task_kwargs)
t1_task_kwargs.mixing_ratio = 1.0 # TODO check which task is T1/T2 in the paper
t2_task_kwargs = copy.deepcopy(gmm_lr_conf.training.task_kwargs)
t2_task_kwargs.mixing_ratio = 0.0

t1_task = get_task_sampler(
    gmm_lr_conf.training.task, n_dims, batch_size, **t1_task_kwargs
)()

t2_task = get_task_sampler(
    gmm_lr_conf.training.task, n_dims, batch_size, **t2_task_kwargs
)()

xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

t1_ys = t1_task.evaluate(xs)
t2_ys = t2_task.evaluate(xs)




with torch.no_grad():
    print('Getting T1 Preds for Mixture Model')
    mix_transformer_t1_preds = gmm_lr_model(xs.to(cuda_device), t1_ys.to(cuda_device)).cpu()
    print('Getting T2 Preds for Mixture Model')
    mix_transformer_t2_preds = gmm_lr_model(xs.to(cuda_device), t2_ys.to(cuda_device)).cpu()




metric = t1_task.get_metric()
mix_transformer_t1_errors = metric(mix_transformer_t1_preds, t1_ys).numpy().squeeze()
mix_transformer_t2_errors = metric(mix_transformer_t2_preds, t2_ys).numpy().squeeze()




def get_df_from_pred_array(pred_arr, n_points, offset=0):
    # pred_arr --> b x pts-1
    batch_size = pred_arr.shape[0]
    flattened_arr = pred_arr.ravel()
    points = np.array(list(range(offset, n_points)) * batch_size)
    df = pd.DataFrame({"y": flattened_arr, "x": points})
    return df


def lineplot_with_ci(pred_or_err_arr, n_points, offset, label, ax, seed):
    sns.lineplot(
        data=get_df_from_pred_array(pred_or_err_arr, n_points=n_points, offset=offset),
        y="y",
        x="x",
        label=label,
        ax=ax,
        n_boot=1000,
        seed=seed,
        ci=90,
    )




sns.set(style="whitegrid", font_scale=1.5)
# latexify(4, 3)

lr_bound = n_dims
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
lineplot_with_ci(
    mix_transformer_t1_errors,
    n_points,
    offset=0,
    label="Transformer (GMM))",
    ax=ax1,
    seed=seed,
)

# ax1.set_xlabel("$k$\n(\# in-context examples)")
# ax1.set_ylabel("$\\texttt{loss@}k$")
# ax1.set_title("Evaluation on $T_1$ Prompts")
ax1.set_xlabel("k (num in-context examples)")
ax1.set_ylabel("loss at k")
ax1.set_title("Evaluation on T_1 Prompts")

ax1.axvline(lr_bound, ls="--", color="black")
ax1.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax1)

lineplot_with_ci(
    mix_transformer_t2_errors,
    n_points,
    offset=0,
    label="Transformer (GMM)",
    ax=ax2,
    seed=seed,
)

# ax2.set_xlabel("$k$\n(\# in-context examples)")
# ax2.set_ylabel("$\\texttt{loss@}k$")
# ax2.set_title("Evaluation on $T_2$ Prompts")
ax2.set_xlabel("k (num in-context examples)")
ax2.set_ylabel("loss at k")
ax2.set_title("Evaluation on T_2 Prompts")

ax2.axvline(lr_bound, ls="--", color="black")
ax2.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax2)

ax1.legend().set_visible(False)
ax2.legend().set_visible(False)

# plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
handles, labels = ax1.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 0.95))
# leg = plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
for line in leg.get_lines():
    line.set_linewidth(5)
plt.savefig(f"plots/gmm_linear_regression_normalize_{gmm_lr_conf.training.task_kwargs.normalize_outputs}.png", dpi=300, bbox_inches="tight")
