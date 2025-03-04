import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_df_from_pred_array(pred_arr, n_points, offset=0):
    # pred_arr --> b x pts-1
    batch_size = pred_arr.shape[0]
    flattened_arr = pred_arr.ravel()
    points = np.array(list(range(offset, n_points)) * batch_size)
    df = pd.DataFrame({"y": flattened_arr, "x": points})
    return df


def lineplot_with_ci(pred_or_err_arr, n_points, offset, label, ax):
    sns.lineplot(
        data=get_df_from_pred_array(pred_or_err_arr, n_points=n_points, offset=offset),
        y="y",
        x="x",
        label=label,
        ax=ax,
        n_boot=1000,
        # errorbar=('ci', 90),
        ci=90,
        linewidth=2,
        legend=False
    )

blue = "#377eb8"
orange = "#ff7f00"
green = "#4daf4a"
m1 = "o"
m2 = "X"
m3 = "^"

layers = [12, 10, 8, 6]
demb = [256, 128, 64, 32]

demb_std = 256
layers_std = 12

run_ids = ["d_256_l_12_h_8", "d_256_l_10_h_8", "d_256_l_8_h_8", "d_256_l_6_h_8"]
# run_ids = ["d_256_l_12_h_8", "d_128_l_12_h_4", "d_64_l_12_h_2", "d_32_l_12_h_1"]
# seed = 45
n_points = 15

latents = [5,15,30]

n_axs = len(latents)

fig, ax = plt.subplots(1, n_axs, figsize=(6*n_axs, 6), sharex=True)

# fig, ax = plt.subplots(1, len(latents), figsize=(48, 15), sharey='all')

for i, true_latent in enumerate(latents):

    for j, run_id in enumerate(run_ids):
        CE_tran = np.load(f'Files/Ablation_v{i+1}_{true_latent}_CE_{run_id}.npy')

        lineplot_with_ci(
            CE_tran,
            n_points,
            offset=0,
            # label=f'demb={demb[j]}',
            label=f'layers={layers[j]}',
            ax=ax[i],
        )

    ax[i].set_title(f"Velocity = {true_latent} m/s, demb={demb_std}")
    # ax[i].set_title(f"Velocity = {true_latent} m/s, layers={layers_std}")
    ax[i].grid()

for ca in ax:
    ca.set_xlabel("# In-context Examples", fontsize=14)
    ca.set_ylabel("Cross entropy", fontsize=14)
    ca.tick_params(axis='both', which='major', labelsize=12)

lines, labels = ax[1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center',bbox_to_anchor=(0.52,-0.15), ncol=3, fontsize=14)

# for line in leg.get_lines():
#     line.set_linewidth(5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"plots/Ablation_style_plot_effect_of_nlayers.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"plots/Ablation_style_plot_effect_of_demb.png", dpi=300, bbox_inches="tight")

print('Done')

# np.save(f'Files/Ablation_v1_5_CE_{run_id}.npy', v1_CE)
# np.save(f'Files/Ablation_v2_15_CE_{run_id}.npy', v2_CE)
# np.save(f'Files/Ablation_v3_30_CE_{run_id}.npy', v3_CE)


# np.save(f'Files/Ablation_v1_5_Acc_{run_id}.npy', v1_acc)
# np.save(f'Files/Ablation_v2_15_Acc_{run_id}.npy', v2_acc)
# np.save(f'Files/Ablation_v3_30_Acc_{run_id}.npy', v3_acc)


# np.save('Files/v1_5_Cond_Ent.npy', v1_Cond_Ent)
# np.save('Files/v2_15_Cond_Ent.npy', v2_Cond_Ent)
# np.save('Files/v3_30_Cond_Ent.npy', v3_Cond_Ent)