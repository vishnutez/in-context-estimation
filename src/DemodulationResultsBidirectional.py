"""
Based on GMMLinearRegressionResults.py.
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
parser.add_argument('task_name', help='Task name for path')
parser.add_argument('--ray_context_file')
parser.add_argument('--fading_context_file')
parser.add_argument('--ray_nocontext_file')
parser.add_argument('--fading_nocontext_file')
parser.add_argument('--step', type=int, default=-1, help='Used to specify a model checkpoint other than the most recent one. -1 means to use the most recent checkpoint.')
# parser.add_argument('--posenc', action='store_true', help='Whether to evaluate the model trained with or without positional encoding')

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
demod_model, demod_conf = get_model_from_run(os.path.join(run_dir, task, run_id), step=args.step)
demod_model.to(cuda_device)





batch_size = 1280  # 1280 #conf.training.batch_size
n_dims = 8
n_points = demod_conf.training.curriculum.points.end
data_sampler = get_data_sampler(demod_conf.training.data, n_dims)





seed = 44
torch.manual_seed(seed)

print(f'normalize_outputs should be false: {demod_conf.training.task_kwargs.normalize_outputs}')

# Create 2 tasks, for the two tasks in the mixture. A hack for doing this is to
# use the same task kwargs but set fading_prob to either 0 or 1.
ray_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
ray_task_kwargs.fading_prob = 0.0
fading_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
fading_task_kwargs.fading_prob = 1.0

ray_task = get_task_sampler(
    demod_conf.training.task, n_dims, batch_size, **ray_task_kwargs
)()

fading_task = get_task_sampler(
    demod_conf.training.task, n_dims, batch_size, **fading_task_kwargs
)()

xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

ray_ys = ray_task.evaluate(xs)
fading_ys = fading_task.evaluate(xs)


with torch.no_grad():
    # For nmask = 0 to n_points - 1, mask out all but the first nmask x's. Compute preds for
    # just the first masked x, to keep in line with previous experiments.
    # x and y are used in the pre-swap sense.
    mix_transformer_ray_preds_list = []
    mix_transformer_fading_preds_list = []

    bsize, _n_points, xdim = xs.shape
    assert _n_points == n_points
    assert ray_ys.shape[0] == fading_ys.shape[0] == bsize
    assert ray_ys.shape[1] == fading_ys.shape[1] == n_points
    ydim = ray_ys.shape[2]
    assert ydim == fading_ys.shape[2]

    for idx_first_mask in range(n_points):
        # modified from train_step()
        mask_indices = np.arange(idx_first_mask, n_points)
        # Masked elements have a nonzero value in the zeroth dim, unmasked elements have a 0.
        y_mask = torch.zeros(bsize, n_points, 1, dtype=ray_ys.dtype, device=ray_ys.device)
        x_mask_single = torch.zeros(1, n_points, 1, dtype=xs.dtype, device=xs.device)
        x_mask_single[0, mask_indices, 0] = 1
        x_mask = torch.tile(x_mask_single, (bsize, 1, 1))
        assert x_mask.shape == (bsize, n_points, 1)

        curr_xs = torch.cat([x_mask, xs], dim=2)
        assert curr_xs.shape == (bsize, n_points, xdim + 1)

        curr_ray_ys = torch.cat([y_mask, ray_ys], dim=2)
        curr_fading_ys = torch.cat([y_mask, fading_ys], dim=2)
        assert curr_ray_ys.shape == curr_fading_ys.shape == (bsize, n_points, ydim + 1)

        # For fair comparison with a similar GPT2 model, fully "mask out" the future (both xs and ys) by
        # removing it before passing tensors to the model.
        curr_xs_sliced = curr_xs[:, :(idx_first_mask + 1), :]
        curr_ray_ys_sliced = curr_ray_ys[:, :(idx_first_mask + 1), :]
        curr_fading_ys_sliced = curr_fading_ys[:, :(idx_first_mask + 1), :]
        assert curr_xs_sliced.shape == (bsize, (idx_first_mask + 1), xdim + 1)
        assert curr_ray_ys_sliced.shape == curr_fading_ys_sliced.shape == (bsize, (idx_first_mask + 1), ydim + 1)

        curr_mix_transformer_ray_preds = demod_model(curr_xs_sliced.to(cuda_device), curr_ray_ys_sliced.to(cuda_device)).cpu()
        curr_mix_transformer_fading_preds = demod_model(curr_xs_sliced.to(cuda_device), curr_fading_ys_sliced.to(cuda_device)).cpu()

        out_dim = curr_mix_transformer_ray_preds.shape[2]
        assert curr_mix_transformer_ray_preds.shape == curr_mix_transformer_fading_preds.shape == (bsize, (idx_first_mask + 1), out_dim)
        # import pdb; pdb.set_trace()
        assert out_dim == xdim

        # Just predict on the one desired element
        curr_mix_transformer_ray_pred_desired = curr_mix_transformer_ray_preds[:, [idx_first_mask], :]  # Keep the inner dimension with size 1
        curr_mix_transformer_fading_pred_desired = curr_mix_transformer_fading_preds[:, [idx_first_mask], :]
        assert curr_mix_transformer_ray_pred_desired.shape == curr_mix_transformer_fading_pred_desired.shape == (bsize, 1, out_dim)

        mix_transformer_ray_preds_list.append(curr_mix_transformer_ray_pred_desired)
        mix_transformer_fading_preds_list.append(curr_mix_transformer_fading_pred_desired)

    mix_transformer_ray_preds = torch.cat(mix_transformer_ray_preds_list, dim=1)
    mix_transformer_fading_preds = torch.cat(mix_transformer_fading_preds_list, dim=1)


metric = ray_task.get_metric()
mix_transformer_ray_errors = metric(mix_transformer_ray_preds, xs).numpy().squeeze()
mix_transformer_fading_errors = metric(mix_transformer_fading_preds, xs).numpy().squeeze()



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
# context = black, nocontext = red
ray_context_values = None
ray_nocontext_values = None
fading_context_values = None
fading_nocontext_values = None
if args.ray_context_file is not None:
    with open(args.ray_context_file) as f:
        ray_context_values = [float(line) for line in f]
if args.ray_nocontext_file is not None:
    with open(args.ray_nocontext_file) as f:
        ray_nocontext_values = [float(line) for line in f]
if args.fading_context_file is not None:
    with open(args.fading_context_file) as f:
        fading_context_values = [float(line) for line in f]
if args.fading_nocontext_file is not None:
    with open(args.fading_nocontext_file) as f:
        fading_nocontext_values = [float(line) for line in f]

get_x_vec = lambda vals: list(range(len(vals)))


sns.set(style="whitegrid", font_scale=0.75)
# latexify(4, 3)

lr_bound = n_dims
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
lineplot_with_ci(
    mix_transformer_ray_errors,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax1,
    seed=seed,
)
if args.ray_context_file is not None:
    sns.lineplot(
        x=get_x_vec(ray_context_values),
        y=ray_context_values,
        color="black",
        label="Known Environment",
        ax=ax1,
    )
if args.ray_nocontext_file is not None:
    sns.lineplot(
        x=get_x_vec(ray_nocontext_values),
        y=ray_nocontext_values,
        color="red",
        label="LMMSE",
        ax=ax1,
    )

# ax1.set_xlabel("$k$\n(\# in-context examples)")
# ax1.set_ylabel("$\\texttt{loss@}k$")
# ax1.set_title("Evaluation on $T_1$ Prompts")
ax1.set_xlabel("k (num in-context examples)")
ax1.set_ylabel("loss@k")
ax1.set_title("Evaluation on Ray Prompts")

ax1.axvline(lr_bound, ls="--", color="black")
ax1.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax1)

lineplot_with_ci(
    mix_transformer_fading_errors,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax2,
    seed=seed,
)
if args.fading_context_file is not None:
    sns.lineplot(
        x=get_x_vec(fading_context_values),
        y=fading_context_values,
        color="black",
        label="Known Environment",
        ax=ax2,
    )
if args.fading_nocontext_file is not None:
    sns.lineplot(
        x=get_x_vec(fading_nocontext_values),
        y=fading_nocontext_values,
        color="red",
        label="LMMSE",
        ax=ax2,
    )

# ax2.set_xlabel("$k$\n(\# in-context examples)")
# ax2.set_ylabel("$\\texttt{loss@}k$")
# ax2.set_title("Evaluation on $T_2$ Prompts")
ax2.set_xlabel("k (num in-context examples)")
ax2.set_ylabel("loss@k")
ax2.set_title("Evaluation on Fading Prompts")

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
numlines = len([file for file in [args.ray_context_file, args.ray_nocontext_file, args.fading_context_file, args.fading_nocontext_file] if file is not None])

plotname = args.task_name
if numlines > 0:
    plotname = plotname + f"_{numlines}lines"
if args.step != -1:
    plotname = plotname + f"_step{args.step}"
plotname = plotname + ".png"
plot_path = os.path.join('plots', plotname)
print("Saving image to", plot_path)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
