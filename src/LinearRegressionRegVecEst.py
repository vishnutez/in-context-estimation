# Only run once!!
import os

from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler

import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor, Ridge
import numpy as np
# import cvxpy
# from cvxpy import Variable, Minimize, Problem
# from cvxpy import norm as cvxnorm

# # from cvxpy import mul_elemwise, SCS
# from cvxpy import vec as cvxvec

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

mpl.rcParams["figure.dpi"] = 300

# mpl.rcParams["text.usetex"] = True

matplotlib.rcParams.update(
    {
        "axes.titlesize": 8,
        "figure.titlesize": 10,  # was 10
        "legend.fontsize": 10,  # was 10
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)
run_dir = "../models"

task = "linear_regression"
run_id = "demb32_nl6_nh1"  # Change according to the id of the model you train
dr_model, dr_conf = get_model_from_run(os.path.join(run_dir, task, run_id))
dr_model.to("cuda:0")

batch_size = 1280  # 1280 #conf.training.batch_size
n_dims = 5
# n_points = dr_conf.training.curriculum.points.end
n_points = 10
data_sampler = get_data_sampler(dr_conf.training.data, n_dims)
task_sampler = get_task_sampler(
    dr_conf.training.task, n_dims, batch_size, **dr_conf.training.task_kwargs
)

seed = 42
torch.manual_seed(seed)
task = task_sampler()
xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)

if task == 'linear_regression_reg_vec_estimation':
    ys, ws = task.evaluate(xs)
    bsize, n_points, n_dims = xs.shape
    # ws is of shape (bsize, n_dims, 1), ws.T is of shape (bsize, 1, n_dims)
    ws_transpose = torch.transpose(ws, -1,-2)
    # print('ws_transpose = ', ws_transpose.shape)
    ws_targets = torch.tile(ws_transpose, (1, n_points, 1))
else:
    ys = task.evaluate(xs)

with torch.no_grad():
    transformer_preds = dr_model(xs.to("cuda:0"), ys.to("cuda:0")).cpu()

metric = task.get_metric()
if task == 'linear_regression_reg_vec_estimation':
    transformer_errors = metric(transformer_preds, ws_targets).numpy().squeeze()
else:
    transformer_errors = metric(transformer_preds, ys).numpy().squeeze()

transformer_mse = np.mean(transformer_errors, axis=(0,)) / n_dims

print("transformer errors = ", transformer_errors.shape)
print("transformer mse = ", transformer_mse)


# def get_df_from_pred_array(pred_arr, n_points, offset=0):
#     # pred_arr --> b x pts-1
#     batch_size = pred_arr.shape[0]
#     flattened_arr = pred_arr.ravel()
#     points = np.array(list(range(offset, n_points)) * batch_size)
#     df = pd.DataFrame({"y": flattened_arr, "x": points})
#     return df


# def lineplot_with_ci(pred_or_err_arr, n_points, offset, label, ax, seed):
#     sns.lineplot(
#         data=get_df_from_pred_array(pred_or_err_arr, n_points=n_points, offset=offset),
#         y="y",
#         x="x",
#         label=label,
#         ax=ax,
#         n_boot=1000,
#         seed=seed,
#         ci=90,
#     )



# sns.set(style="whitegrid", font_scale=1.5)
# # latexify(4, 3)
bound = n_dims
# fig, ax = plt.subplots()
# ax.plot(list(range(n_points)), transformer_pe_errors.mean(axis=0), label = "With Position Encodings")
# ax.plot(list(range(n_points)), transformer_no_pe_errors.mean(axis=0), label = "Without Position Encodings")
# lineplot_with_ci(
#     transformer_mse,
#     n_points,
#     offset=0,
#     label="Transformer",
#     ax=ax,
#     seed=seed,
# )

# lineplot_with_ci(lsq_errors / n_dims, n_points, offset=0, label="OLS", ax=ax, seed=seed)
# lineplot_with_ci(
#     ridge_errors / n_dims, n_points, offset=0, label="Ridge (0.01)", ax=ax, seed=seed
# )
# lineplot_with_ci(l2_norm_errors, n_points, label="L-2 Norm Min", ax=ax, seed=seed)

plt.plot(transformer_mse, color='darkred', lw=2, label='nl = 6 demb = 32 nh = 1')
plt.xlabel("# in-context examples")
plt.ylabel("mean squared error")
plt.title("Dense Regression ICL")
plt.axvline(bound, ls="--", color="black")
plt.annotate("bound", xy=(bound + 0.5, 0.6), color="darkgrey", rotation=0)
# format_axes(ax)
# plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
plt.legend()
plt.savefig("plots/linear_regression_d5_nh1_demb32_nl6.png")
plt.show()