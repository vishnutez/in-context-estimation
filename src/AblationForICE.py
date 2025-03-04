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

# # Only used for --save_predictions and --load_predictions
# params_dict_file = os.path.join('plots', f'params_dict_{task}_{run_id}.pickle')
# v1_errors_file = os.path.join('plots', f'v1_errors_{task}_{run_id}.npy')
# v2_errors_file = os.path.join('plots', f'v2_errors_{task}_{run_id}.npy')
# v3_errors_file = os.path.join('plots', f'v3_errors_{task}_{run_id}.npy')

batch_size = 5000  # 1280 #conf.training.batch_size
n_dims = 8
seed = 45

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

xs, sig_ids = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

v1_ys = v1_task.evaluate(xs)
v2_ys = v2_task.evaluate(xs)
v3_ys = v3_task.evaluate(xs)

with torch.no_grad():
    print('Getting v1 Preds for Mixture Model')
    v1_pred_logits = demod_model(xs.to(cuda_device), v1_ys.to(cuda_device)).cpu()
    print('Getting v2 Preds for Mixture Model')
    v2_pred_logits = demod_model(xs.to(cuda_device), v2_ys.to(cuda_device)).cpu()
    print('Getting v3 Preds for Mixture Model')
    v3_pred_logits = demod_model(xs.to(cuda_device), v3_ys.to(cuda_device)).cpu()

    v1_pred_sig_ids = torch.argmax(v1_pred_logits, dim=-1).detach().numpy()
    v2_pred_sig_ids = torch.argmax(v2_pred_logits, dim=-1).detach().numpy()
    v3_pred_sig_ids = torch.argmax(v3_pred_logits, dim=-1).detach().numpy()

metric = v1_task.get_metric()


v1_post = torch.softmax(v1_pred_logits, dim=-1).detach().numpy()
v2_post = torch.softmax(v2_pred_logits, dim=-1).detach().numpy()
v3_post = torch.softmax(v3_pred_logits, dim=-1).detach().numpy()

s_ids_np = sig_ids.detach().numpy()

batch_size, cnxt_len = s_ids_np.shape

B_indices = np.arange(batch_size)[:, None]


v1_Cond_Ent = np.mean(-v1_post * np.log(v1_post), axis=-1)
v2_Cond_Ent = np.mean(-v2_post * np.log(v2_post), axis=-1)
v3_Cond_Ent = np.mean(-v3_post * np.log(v3_post), axis=-1)


v1_CE = -np.log(v1_post[B_indices, np.arange(cnxt_len), s_ids_np])
v2_CE = -np.log(v2_post[B_indices, np.arange(cnxt_len), s_ids_np])
v3_CE = -np.log(v3_post[B_indices, np.arange(cnxt_len), s_ids_np])

v1_acc = (v1_pred_sig_ids == s_ids_np) * 100
v2_acc = (v2_pred_sig_ids == s_ids_np) * 100
v3_acc = (v3_pred_sig_ids == s_ids_np) * 100


print('v1_CE = ', v1_CE.shape)



# v1_errors = metric(v1_preds, xs).numpy().squeeze()
# v2_errors = metric(v2_preds, xs).numpy().squeeze()
# v3_errors = metric(v3_preds, xs).numpy().squeeze()

#     assert v1_errors.shape == v2_errors.shape == v3_errors.shape == (batch_size, n_points)
#     if args.shorten_1:
#         n_points = n_points - 1
#         v1_errors = v1_errors[:, :-1]
#         v2_errors = v2_errors[:, :-1]
#         v3_errors = v3_errors[:, :-1]
#     assert v1_errors.shape == v2_errors.shape == v3_errors.shape == (batch_size, n_points)

#     if args.save_predictions:
#         params_dict = {
#             'n_points': n_points,
#         }
#         with open(params_dict_file, mode='wb') as f:
#             pickle.dump(params_dict, f, pickle.HIGHEST_PROTOCOL)
        
#         np.save(v1_errors_file, v1_errors, allow_pickle=False)
#         np.save(v2_errors_file, v2_errors, allow_pickle=False)
#         np.save(v3_errors_file, v3_errors, allow_pickle=False)
# else:
#     with open(params_dict_file, mode='rb') as f:
#         params_dict = pickle.load(f)
#     n_points = params_dict['n_points']

#     v1_errors = np.load(v1_errors_file)
#     v2_errors = np.load(v2_errors_file)
#     v3_errors = np.load(v3_errors_file)

# if args.genie_file is not None:
#     genie_df = pd.read_csv(args.genie_file)

# if args.lmmse_file is not None:
#     lmmse_df = pd.read_csv(args.lmmse_file)

# get_x_vec = lambda vals: list(range(len(vals)))


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
ax1.set_ylim(-5, 105)
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
lineplot_with_ci(
    v1_acc,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax1,
    seed=seed,
)

# ax1.set_xlabel("$k$\n(\# in-context examples)")
# ax1.set_ylabel("$\\texttt{loss@}k$")
# ax1.set_title("Evaluation on $T_1$ Prompts")
ax1.set_xlabel("Context length")
ax1.set_ylabel("SEP")
ax1.set_title("Velocity = 5 m/s")

# ax1.axvline(lr_bound, ls="--", color="black")
# ax1.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax1)

lineplot_with_ci(
    v2_acc,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax2,
    seed=seed,
)

# ax2.set_xlabel("$k$\n(\# in-context examples)")
# ax2.set_ylabel("$\\texttt{loss@}k$")
# ax2.set_title("Evaluation on $T_2$ Prompts")
ax2.set_xlabel("Context length")
ax2.set_ylabel("SEP")
ax2.set_title("Velocity = 15 m/s")

# ax2.axvline(lr_bound, ls="--", color="black")
# ax2.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax2)


lineplot_with_ci(
    v3_acc,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax3,
    seed=seed,
)


ax3.set_xlabel("Context length")
ax3.set_ylabel("SEP")
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



plt.savefig(f"plots/{args.task_name}_{args.run_id}.png", dpi=300, bbox_inches="tight")


np.save(f'Files/Ablation_v1_5_CE_{run_id}.npy', v1_CE)
np.save(f'Files/Ablation_v2_15_CE_{run_id}.npy', v2_CE)
np.save(f'Files/Ablation_v3_30_CE_{run_id}.npy', v3_CE)


np.save(f'Files/Ablation_v1_5_Acc_{run_id}.npy', v1_acc)
np.save(f'Files/Ablation_v2_15_Acc_{run_id}.npy', v2_acc)
np.save(f'Files/Ablation_v3_30_Acc_{run_id}.npy', v3_acc)


# np.save('Files/v1_5_Cond_Ent.npy', v1_Cond_Ent)
# np.save('Files/v2_15_Cond_Ent.npy', v2_Cond_Ent)
# np.save('Files/v3_30_Cond_Ent.npy', v3_Cond_Ent)