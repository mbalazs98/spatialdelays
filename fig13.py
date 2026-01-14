import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import os
import re

sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
seeds = 20
pattern = re.compile(r"^(\d+)-")
fig, ax = plt.subplots(1,1,figsize=(3.25, 2))



no_space = []
no_cost = []
space_cost = []


no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-10_0.05_1e-10_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-11_0.05_1e-09_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-11_0.05_1e-08_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_")


def shannon_entropy_comms(W):
    W = np.abs(W)
    N = W.shape[0]
    s = np.diag(np.sum(W, axis=1))
    s_inv_sqrt = np.zeros_like(s)
    nonzero_mask = np.diag(s) > 0
    s_inv_sqrt[nonzero_mask, nonzero_mask] = 1.0 / np.sqrt(np.diag(s)[nonzero_mask])
    adj = s_inv_sqrt @ W @ s_inv_sqrt
    ncabs = expm(adj)
    row_sums = ncabs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = ncabs / row_sums
    P_safe = np.where(P > 0, P, 1)
    H_rows = -np.sum(P_safe * np.log2(P_safe), axis=1)
    se_ncabs = np.mean(H_rows)
    return se_ncabs

no_space_ent = []
no_space_ent_std = []
no_cost_ent = []
no_cost_ent_std = []
space_cost_ent = []
space_cost_ent_std = []
for el in no_space:
    ent = []
    for seed in range(seeds):
        last_epoch = 0

        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    no_space_ent.append(np.mean(ent))
    no_space_ent_std.append(np.std(ent))
print(f'Final shannon of no space network: {no_space_ent[-1]}\n')


label = "Non-spatial"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=no_space_ent, yerr=no_space_ent_std, marker="o", markersize=4,
            color=palette[1], label=label, capsize=2, linewidth=0.7)


for el in no_cost:
    ent = []
    for seed in range(seeds):
        last_epoch = 0

        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    no_cost_ent.append(np.mean(ent))
    no_cost_ent_std.append(np.std(ent))
print(f'Final shannon of no cost network: {no_cost_ent[-1]}\n')

label = "Spatial"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=no_cost_ent, yerr=no_cost_ent_std, marker="o", markersize=4,
            color=palette[0], label=label, capsize=2, linewidth=0.7)


for el in space_cost:
    ent = []
    for seed in range(seeds):
        last_epoch = 0

        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    space_cost_ent.append(np.mean(ent))
    space_cost_ent_std.append(np.std(ent))
print(f'Final shannon of space cost network: {space_cost_ent[-1]}\n')

label = "Spatial+cost"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=space_cost_ent, yerr=space_cost_ent_std, marker="o", markersize=4,
            color=palette[3], label=label, capsize=2, linewidth=0.7)



ax.set_xlabel('Regularisation')
ax.set_ylabel('Communicability entropy')


ax.set_xscale("log")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
ax.set_xticks([1e-10, 1e-09, 1e-08])
titles = [
    "   Weak" + r"$\rightarrow$",
    "   Medium" + r"$\rightarrow$",
    "Strong"
]
ax.set_xticklabels(titles)
plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.subplots_adjust(top=0.93)

plt.savefig("shd_cent.pdf") #76.19, 76.68, 79.20
