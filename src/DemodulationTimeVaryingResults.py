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
parser.add_argument('--genie_file', help='CSV file with 3 columns')
parser.add_argument('--lmmse_file', help='CSV file with 3 columns')
parser.add_argument('--shorten_1', action='store_true', help='Whether to truncate sequences by removing the last idx to match with the genie/lmmse files')  # TODO generalize to any int
parser.add_argument('--save_predictions', action='store_true', help='If given, will save model outputs (numpy arrays) + task parameters (pickled dict) in the "plots" directory with filenames given by run_id and task_name.')
parser.add_argument('--load_predictions', action='store_true', help='If given, will load model outputs previously saved using --save_predictions option. In this case, the model itself will NOT be loaded and a GPU is not required to produce plots.')
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


# Only used for --save_predictions and --load_predictions
params_dict_file = os.path.join('plots', f'params_dict_{task}_{run_id}.pickle')
v1_errors_file = os.path.join('plots', f'v1_errors_{task}_{run_id}.npy')
v2_errors_file = os.path.join('plots', f'v2_errors_{task}_{run_id}.npy')
v3_errors_file = os.path.join('plots', f'v3_errors_{task}_{run_id}.npy')

batch_size = 5000  # 1280 #conf.training.batch_size
n_dims = 8
seed = 45

if not args.load_predictions:
    demod_model, demod_conf = get_model_from_run(os.path.join(run_dir, task, run_id))
    demod_model.to(cuda_device)





    n_points = demod_conf.training.curriculum.points.end
    data_sampler = get_data_sampler(demod_conf.training.data, n_dims)





    torch.manual_seed(seed)


    # Create 3 tasks, for the three tasks in the mixture.
    v1_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
    v1_task_kwargs.v_probs = [1.0, 0.0, 0.0]
    v2_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
    v2_task_kwargs.v_probs = [0.0, 1.0, 0.0]
    v3_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
    v3_task_kwargs.v_probs = [0.0, 0.0, 1.0]

    v1_task = get_task_sampler(
        demod_conf.training.task, n_dims, batch_size, **v1_task_kwargs
    )()

    v2_task = get_task_sampler(
        demod_conf.training.task, n_dims, batch_size, **v2_task_kwargs
    )()

    v3_task = get_task_sampler(
        demod_conf.training.task, n_dims, batch_size, **v3_task_kwargs
    )()

    xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

    v1_ys = v1_task.evaluate(xs)
    v2_ys = v2_task.evaluate(xs)
    v3_ys = v3_task.evaluate(xs)





    with torch.no_grad():
        print('Getting v1 Preds for Mixture Model')
        v1_preds = demod_model(xs.to(cuda_device), v1_ys.to(cuda_device)).cpu()
        print('Getting v2 Preds for Mixture Model')
        v2_preds = demod_model(xs.to(cuda_device), v2_ys.to(cuda_device)).cpu()
        print('Getting v3 Preds for Mixture Model')
        v3_preds = demod_model(xs.to(cuda_device), v3_ys.to(cuda_device)).cpu()



    metric = v1_task.get_metric()
    v1_errors = metric(v1_preds, xs).numpy().squeeze()
    v2_errors = metric(v2_preds, xs).numpy().squeeze()
    v3_errors = metric(v3_preds, xs).numpy().squeeze()

    assert v1_errors.shape == v2_errors.shape == v3_errors.shape == (batch_size, n_points)
    if args.shorten_1:
        n_points = n_points - 1
        v1_errors = v1_errors[:, :-1]
        v2_errors = v2_errors[:, :-1]
        v3_errors = v3_errors[:, :-1]
    assert v1_errors.shape == v2_errors.shape == v3_errors.shape == (batch_size, n_points)

    if args.save_predictions:
        params_dict = {
            'n_points': n_points,
        }
        with open(params_dict_file, mode='wb') as f:
            pickle.dump(params_dict, f, pickle.HIGHEST_PROTOCOL)
        
        np.save(v1_errors_file, v1_errors, allow_pickle=False)
        np.save(v2_errors_file, v2_errors, allow_pickle=False)
        np.save(v3_errors_file, v3_errors, allow_pickle=False)
else:
    with open(params_dict_file, mode='rb') as f:
        params_dict = pickle.load(f)
    n_points = params_dict['n_points']

    v1_errors = np.load(v1_errors_file)
    v2_errors = np.load(v2_errors_file)
    v3_errors = np.load(v3_errors_file)

if args.genie_file is not None:
    genie_df = pd.read_csv(args.genie_file)

if args.lmmse_file is not None:
    lmmse_df = pd.read_csv(args.lmmse_file)

get_x_vec = lambda vals: list(range(len(vals)))


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


sns.set(style="whitegrid", font_scale=2.5)
# latexify(4, 3)

blue = "#377eb8"
orange = "#ff7f00"
green = "#4daf4a"
m1 = "o"
m2 = "X"
m3 = "^"

lr_bound = n_dims
# Correct aspect ratio to use: width 4.8 to height 1.5.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(48, 15), constrained_layout=True, sharey='all')
# ax1.set_ylim(0.75, 2.3)
ax1.set_ylim(0.3, 2.3)
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
lineplot_with_ci(
    v1_errors,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax1,
    seed=seed,
)

if args.genie_file is not None:
    sns.lineplot(
        x=get_x_vec(genie_df["0"]),
        y=genie_df["0"],
        color=green,
        marker=m2,
        label="Context-Aware Estimator",
        ax=ax1,
        linewidth=5,
        markersize=22,
    )

if args.lmmse_file is not None:
    sns.lineplot(
        x=get_x_vec(lmmse_df["0"]),
        y=lmmse_df["0"],
        color=orange,
        marker=m3,
        label="Context-Agnostic Baseline",
        ax=ax1,
        linewidth=5,
        markersize=22,
    )

# ax1.set_xlabel("$k$\n(\# in-context examples)")
# ax1.set_ylabel("$\\texttt{loss@}k$")
# ax1.set_title("Evaluation on $T_1$ Prompts")
ax1.set_xlabel("Context Length")
ax1.set_ylabel("MSE")
ax1.set_title("Velocity = 5 m/s")

# ax1.axvline(lr_bound, ls="--", color="black")
# ax1.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax1)

lineplot_with_ci(
    v2_errors,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax2,
    seed=seed,
)

if args.genie_file is not None:
    sns.lineplot(
        x=get_x_vec(genie_df["1"]),
        y=genie_df["1"],
        color=green,
        marker=m2,
        label="Context-Aware Estimator",
        ax=ax2,
        linewidth=5,
        markersize=22,
    )

if args.lmmse_file is not None:
    sns.lineplot(
        x=get_x_vec(lmmse_df["1"]),
        y=lmmse_df["1"],
        color=orange,
        marker=m3,
        label="Context-Agnostic Baseline",
        ax=ax2,
        linewidth=5,
        markersize=22,
    )

# ax2.set_xlabel("$k$\n(\# in-context examples)")
# ax2.set_ylabel("$\\texttt{loss@}k$")
# ax2.set_title("Evaluation on $T_2$ Prompts")
ax2.set_xlabel("Context Length")
ax2.set_ylabel("MSE")
ax2.set_title("Velocity = 15 m/s")

# ax2.axvline(lr_bound, ls="--", color="black")
# ax2.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax2)


lineplot_with_ci(
    v3_errors,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax3,
    seed=seed,
)

if args.genie_file is not None:
    sns.lineplot(
        x=get_x_vec(genie_df["2"]),
        y=genie_df["2"],
        color=green,
        marker=m2,
        label="Context-Aware Estimator",
        ax=ax3,
        linewidth=5,
        markersize=22,
    )

if args.lmmse_file is not None:
    sns.lineplot(
        x=get_x_vec(lmmse_df["2"]),
        y=lmmse_df["2"],
        color=orange,
        marker=m3,
        label="Context-Agnostic Baseline",
        ax=ax3,
        linewidth=5,
        markersize=22,
    )


ax3.set_xlabel("Context Length")
ax3.set_ylabel("MSE")
ax3.set_title("Velocity = 30 m/s")

# ax3.axvline(lr_bound, ls="--", color="black")
# ax3.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax3)

leg = ax1.legend(loc="upper right")
leg = ax2.legend(loc="upper right")
leg = ax3.legend(loc="upper right")
# ax2.legend().set_visible(False)
# ax3.legend().set_visible(False)

for line in leg.get_lines():
    line.set_linewidth(5)

plt.savefig(f"plots/{args.task_name}.png", dpi=300, bbox_inches="tight")
