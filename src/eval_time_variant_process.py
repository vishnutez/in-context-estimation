"""
Based on GMMLinearRegressionResults.py.
When running this script, the current directory should be 'src'.
Outputs will be saved to 'src/plots'.
"""

from collections import OrderedDict
import re
import os
import sys
import copy
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler


import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor, Ridge
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run_id', help='Run ID (directory name) for the desired run')
parser.add_argument('task_name', help='Task name for path')
parser.add_argument('--seed', type=int, default=45, help='Random seed')
parser.add_argument('--batch_size', type=int, default=1280, help='Batch size')

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

task = args.task_name
run_id = args.run_id


model, config = get_model_from_run(os.path.join(run_dir, task, run_id))
model.to(cuda_device)

n_dims = config.model.n_dims
n_points = config.training.curriculum.points.end
data_sampler = get_data_sampler(config.training.data, n_dims)

torch.manual_seed(args.seed)

v_list_str = config.training.task_kwargs.v_list.split(" ")

v_list = [float(v) for v in v_list_str]

context_task_config = []
for idx in range(len(v_list)):
    task_kwargs = copy.deepcopy(config.training.task_kwargs)
    task_kwargs.v_probs = [0.0] * len(v_list)
    task_kwargs.v_probs[idx] = 1.0
    context_task_config.append(task_kwargs)

print(v_list)

context_task = [get_task_sampler(config.training.task, n_dims, args.batch_size, **v_task)() for v_task in context_task_config]

x, s = data_sampler.sample_xs(b_size=args.batch_size, n_points=n_points)

y = [v_task.evaluate(x) for v_task in context_task]  # y0, y1, y2


tf_logits = []
tf_post = []
tf_map = []
tf_acc = []
tf_ce = []

batch_idxs = np.arange(args.batch_size)[:, None]
for idx in range(len(v_list)):
    with torch.no_grad():
        tf_logits.append(model(x.to(cuda_device), y [idx].to(cuda_device)).cpu())
    tf_map.append(torch.argmax(tf_logits[idx], dim=-1).detach().numpy())  # MAP estimate
    tf_post.append(torch.softmax(tf_logits[idx], dim=-1).detach().numpy())
    tf_acc.append((tf_map[idx] == s).detach().numpy() * 100)
    tf_ce.append(-np.log(tf_post[idx][batch_idxs, np.arange(n_points), s.detach().numpy()]))


def get_df_from_pred_array(pred_arr, n_points, offset=0):
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
        color=blue,
        marker=m1,
        linewidth=5,
        markersize=22,
    )


sns.set(style="whitegrid", font_scale=2.5)

blue = "#377eb8"
orange = "#ff7f00"
green = "#4daf4a"
m1 = "o"
m2 = "X"
m3 = "^"

fig, axs = plt.subplots(1, 3, figsize=(48, 15), constrained_layout=True, sharey='all')

for i, ax in enumerate(axs):
    
    lineplot_with_ci(
        tf_ce[i],
        n_points,
        offset=0,
        label="TF",
        ax=ax,
        seed=args.seed,
    )
    ax.set_xlabel("Context length")
    ax.set_ylabel("SEP")
    ax.set_title(f"Velocity = {v_list[i]} m/s")
    format_axes(ax)
    leg = ax.legend(loc="upper right")


for line in leg.get_lines():
    line.set_linewidth(5)

snr_db = config.training.task_kwargs.snr
if "qam" in config.training.data:
    constellation_type = f"{config.training.data_kwargs.M}qam"
else:
    constellation_type = "qpsk"

plt.savefig(f"../plots/ce_tf_time_var_{constellation_type}_snr_db{snr_db}.png", dpi=300, bbox_inches="tight")

for i, v in enumerate(v_list):
    np.save(f'../results/ce_tf_time_var_v{v}_{constellation_type}_snr_db{snr_db}.npy', tf_ce[i])

print("Done!")