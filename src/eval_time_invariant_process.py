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

from eval import get_model_from_run
# from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler
# from tasks import LinearRegression as LinearRegressionTask


import matplotlib as mpl
# from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor, Ridge
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run_id', help='Run ID (directory name) for the desired run')
parser.add_argument('task_name', help='Task name for path')
parser.add_argument('--seed', type=int, default=43, help='Random seed')
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

snr_db = config.training.task_kwargs.snr
if "qam" in config.training.data:
    constellation_type = f"{config.training.data_kwargs.M}qam"
else:
    constellation_type = "qpsk"

torch.manual_seed(args.seed)

print(f'Normalize_outputs should be false: {config.training.task_kwargs.normalize_outputs}')

# Create 2 tasks, for the two tasks in the mixture. A hack for doing this is to
# use the same task kwargs but set fading_prob to either 0 or 1.

ray_task_kwargs = copy.deepcopy(config.training.task_kwargs)
ray_task_kwargs.fading_prob = 0.0
fading_task_kwargs = copy.deepcopy(config.training.task_kwargs)
fading_task_kwargs.fading_prob = 1.0

ray_task = get_task_sampler(
    config.training.task, n_dims, args.batch_size, **ray_task_kwargs
)()

fading_task = get_task_sampler(
    config.training.task, n_dims, args.batch_size, **fading_task_kwargs
)()

x, s = data_sampler.sample_xs(b_size=args.batch_size, n_points=n_points)

y_ray = ray_task.evaluate(x)
y_fading = fading_task.evaluate(x)

with torch.no_grad():
    print('Getting Ray Preds for Mixture Model')
    tf_logits_ray = model(x.to(cuda_device), y_ray.to(cuda_device)).cpu()
    print('Getting Fading Preds for Mixture Model')
    tf_logits_fading = model(x.to(cuda_device), y_fading.to(cuda_device)).cpu()


metric = ray_task.get_metric()

tf_ray_map = torch.argmax(tf_logits_ray, dim=-1)  # MAP estimate
tf_fading_map = torch.argmax(tf_logits_fading, dim=-1)  

tf_ray_post = torch.softmax(tf_logits_ray, dim=-1).detach().numpy()
tf_fading_post = torch.softmax(tf_logits_fading, dim=-1).detach().numpy()

args.batch_size = len(tf_ray_post)

s_np = s.detach().numpy()

args.batch_size, context_len = s_np.shape
batch_idxs = np.arange(args.batch_size)[:, None]

tf_ray_ce =  -np.log(tf_ray_post[batch_idxs, np.arange(context_len), s_np])
tf_fading_ce = -np.log(tf_fading_post[batch_idxs, np.arange(context_len), s_np])

tf_ray_acc = (((tf_ray_map == s) * 1.0) * 100).detach().numpy()
tf_fading_acc = (((tf_fading_map == s) * 1.0) * 100).detach().numpy()

np.save(f'../results/acc_tf_ray_{constellation_type}_snr_db{snr_db}.npy', tf_ray_acc)
np.save(f'../results/acc_tf_fading_{constellation_type}_snr_db{snr_db}.npy', tf_fading_acc)

np.save(f'../results/ce_tf_ray_{constellation_type}_snr_db{snr_db}.npy', tf_ray_ce)
np.save(f'../results/ce_tf_fading_{constellation_type}_snr_db{snr_db}.npy', tf_fading_ce)

blue = "#377eb8"
orange = "#ff7f00"
green = "#4daf4a"
m1 = "o"
m2 = "X"
m3 = "^"

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
        color=blue,
        marker=m1,
        linewidth=5,
        markersize=22,
    )

get_x_vec = lambda vals: list(range(len(vals)))

sns.set(style="whitegrid", font_scale=2.5)

lr_bound = n_dims
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 15), constrained_layout=True, sharey='all')

lineplot_with_ci(
    tf_ray_ce,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax1,
    seed=args.seed,
)
ax1.set_xlabel("Context Length")
ax1.set_ylabel("MSE")
ax1.set_title(f"1-Ray Channel Model, SNR = {snr_db} dB")
format_axes(ax1)

lineplot_with_ci(
    tf_fading_ce,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax2,
    seed=args.seed,
)
ax2.set_xlabel("Context Length")
ax2.set_ylabel("MSE")
ax2.set_title(f"Rich-Scattering Channel Model, SNR = {snr_db} dB")
format_axes(ax2)

leg = ax1.legend(loc='upper right')
leg = ax2.legend(loc='upper right')
# ax2.legend().set_visible(False)

for line in leg.get_lines():
    line.set_linewidth(5)
plt.savefig(f"../plots/ce_tf_time_inv_{constellation_type}_snr_db{snr_db}.png", dpi=300, bbox_inches="tight")

print('Done')