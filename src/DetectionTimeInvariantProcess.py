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
# from plot_utils import basic_plot, collect_results, relevant_model_names
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
parser.add_argument('--shorten_1', action='store_true', help='Whether to truncate sequences by removing the last idx to match with the ray/fading files')  # TODO generalize to any int
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
ray_errors_file = os.path.join('plots', f'ray_errors_{task}_{run_id}.npy')
fading_errors_file = os.path.join('plots', f'fading_errors_{task}_{run_id}.npy')

batch_size = 1280  # 1280 #conf.training.batch_size
n_dims = 8
seed = 43

# if not args.load_predictions:
demod_model, demod_conf = get_model_from_run(os.path.join(run_dir, task, run_id))
demod_model.to(cuda_device)


n_points = demod_conf.training.curriculum.points.end
data_sampler = get_data_sampler(demod_conf.training.data, n_dims)


torch.manual_seed(seed)

print(f'normalize_outputs should be false: {demod_conf.training.task_kwargs.normalize_outputs}')

# Create 2 tasks, for the two tasks in the mixture. A hack for doing this is to
# use the same task kwargs but set fading_prob to either 0 or 1.
ray_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
ray_task_kwargs.fading_prob = 0.0
fading_task_kwargs = copy.deepcopy(demod_conf.training.task_kwargs)
fading_task_kwargs.fading_prob = 1.0
snr = int(ray_task_kwargs['snr'])

ray_task = get_task_sampler(
    demod_conf.training.task, n_dims, batch_size, **ray_task_kwargs
)()

fading_task = get_task_sampler(
    demod_conf.training.task, n_dims, batch_size, **fading_task_kwargs
)()

xs, s_ids = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

ray_ys = ray_task.evaluate(xs)
fading_ys = fading_task.evaluate(xs)




with torch.no_grad():
    print('Getting Ray Preds for Mixture Model')
    mix_transformer_ray_logits = demod_model(xs.to(cuda_device), ray_ys.to(cuda_device)).cpu()
    print('Getting Fading Preds for Mixture Model')
    mix_transformer_fading_logits = demod_model(xs.to(cuda_device), fading_ys.to(cuda_device)).cpu()


metric = ray_task.get_metric()

mix_tf_ray_sym_preds = torch.argmax(mix_transformer_ray_logits, dim=-1)
mix_tf_fading_sym_preds = torch.argmax(mix_transformer_fading_logits, dim=-1)  



tf_ray_post = torch.softmax(mix_transformer_ray_logits, dim=-1).detach().numpy()
tf_fading_post = torch.softmax(mix_transformer_fading_logits, dim=-1).detach().numpy()

batch_size = len(tf_ray_post)

s_ids_np = s_ids.detach().numpy()

batch_size, cnxt_len = s_ids_np.shape

B_indices = np.arange(batch_size)[:, None]

tf_ray_CE =  -np.log(tf_ray_post[B_indices, np.arange(cnxt_len), s_ids_np])
tf_fading_CE = -np.log(tf_fading_post[B_indices, np.arange(cnxt_len), s_ids_np])

tf_accuracy_one_ray = (((mix_tf_ray_sym_preds == s_ids)*1.0) * 100).detach().numpy()
tf_accuracy_fading = (((mix_tf_fading_sym_preds == s_ids)*1.0) * 100).detach().numpy()

tf_ray_Cond_Ent = np.mean(-tf_ray_post * np.log(tf_ray_post), axis=-1)
tf_fading_Cond_Ent = np.mean(-tf_fading_post * np.log(tf_fading_post), axis=-1)


print('mix tf ray errors = ', tf_ray_CE.shape)



# assert mix_transformer_ray_errors.shape == mix_transformer_fading_errors.shape == (batch_size, n_points)
# if args.shorten_1:
#     n_points = n_points - 1
#     mix_transformer_ray_errors = mix_transformer_ray_errors[:, :-1]
#     mix_transformer_fading_errors = mix_transformer_fading_errors[:, :-1]
# assert mix_transformer_ray_errors.shape == mix_transformer_fading_errors.shape == (batch_size, n_points)

if args.save_predictions:
    params_dict = {
        'n_points': n_points,
        'snr': snr,
    }
    with open(params_dict_file, mode='wb') as f:
        pickle.dump(params_dict, f, pickle.HIGHEST_PROTOCOL)
    
    np.save(ray_errors_file, tf_ray_CE, allow_pickle=False)
    np.save(fading_errors_file, tf_fading_CE, allow_pickle=False)


np.save(f'../results/Accuracy_one_ray_{task}.npy', tf_accuracy_one_ray)
np.save(f'../results/Accuracy_fading_{task}.npy', tf_accuracy_fading)

np.save(f'../results/CE_one_ray_{task}.npy', tf_ray_CE)
np.save(f'../results/CE_fading_{task}.npy', tf_fading_CE)

# else:
    # with open(params_dict_file, mode='rb') as f:s
    #     params_dict = pickle.load(f)
    # n_points = params_dict['n_points']
    # snr = params_dict['snr']

    # mix_transformer_ray_errors = np.load(ray_errors_file)
    # mix_transformer_fading_errors = np.load(fading_errors_file)


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


sns.set(style="whitegrid", font_scale=2.5)
# latexify(4, 3)

# plot cross-entropy

lr_bound = n_dims
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 15), constrained_layout=True, sharey='all')
# ax1.set_ylim(0.75, 2.3)
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
lineplot_with_ci(
    tf_ray_CE,
    n_points,
    offset=0,
    label="Transformer",
    ax=ax1,
    seed=seed,
)
# if args.ray_context_file is not None:
#     sns.lineplot(
#         x=get_x_vec(ray_context_values),
#         y=ray_context_values,
#         color=green,
#         marker=m2,
#         label="Context-Aware Estimator",
#         ax=ax1,
#         linewidth=5,
#         markersize=22,
#     )
# if args.ray_nocontext_file is not None:
#     sns.lineplot(
#         x=get_x_vec(ray_nocontext_values),
#         y=ray_nocontext_values,
#         color=orange,
#         marker=m3,
#         label="Context-Agnostic Baseline",
#         ax=ax1,
#         linewidth=5,
#         markersize=22,
#     )

# ax1.set_xlabel("$k$\n(\# in-context examples)")
# ax1.set_ylabel("$\\texttt{loss@}k$")
# ax1.set_title("Evaluation on $T_1$ Prompts")
ax1.set_xlabel("Context Length")
ax1.set_ylabel("MSE")
ax1.set_title(f"1-Ray Channel Model, SNR = {snr} dB")

# ax1.axvline(lr_bound, ls="--", color="black")
# ax1.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax1)

lineplot_with_ci(
    tf_fading_CE,
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
        color=green,
        marker=m2,
        label="Context-Aware Estimator",
        ax=ax2,
        linewidth=5,
        markersize=22,
    )
if args.fading_nocontext_file is not None:
    sns.lineplot(
        x=get_x_vec(fading_nocontext_values),
        y=fading_nocontext_values,
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
ax2.set_title(f"Rich-Scattering Channel Model, SNR = {snr} dB")

# ax2.axvline(lr_bound, ls="--", color="black")
# ax2.annotate("Bound", xy=(lr_bound + 0.25, 0.5), color="r", rotation=0)
format_axes(ax2)

leg = ax1.legend(loc='upper right')
leg = ax2.legend(loc='upper right')
# ax2.legend().set_visible(False)

for line in leg.get_lines():
    line.set_linewidth(5)
numlines = len([file for file in [args.ray_context_file, args.ray_nocontext_file, args.fading_context_file, args.fading_nocontext_file] if file is not None])
plt.savefig(f"../plots/{args.task_name}_{numlines}_cross_entropy.png", dpi=300, bbox_inches="tight")

print('Done')