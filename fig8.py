import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
seeds = 20
fig, ax = plt.subplots(4,1,figsize=(3.25, 5.5))


no_space = []
no_cost = []
space_cost = []


no_space.append("results_pos/prune_nolimit_128_0_5e-10_0.05_1e-10_0_")
no_cost.append("results_pos/prune_nolimit_128_2_5e-10_0.05_1e-10_0_")
space_cost.append("results_pos/prune_nolimit_128_2_5e-10_0.05_1e-10_2_")

no_space.append("results_pos/prune_nolimit_128_0_5e-11_0.05_1e-09_0_")
no_cost.append("results_pos/prune_nolimit_128_2_5e-10_0.05_1e-09_0_")
space_cost.append("results_pos/prune_nolimit_128_2_5e-10_0.05_1e-09_2_")

no_space.append("results_pos/prune_nolimit_128_0_5e-11_0.05_1e-08_0_")
no_cost.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-08_0_")
space_cost.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-08_2_")


no_space.append("results_pos/prune_nolimit_128_0_5e-13_0.05_1e-07_0_")
no_cost.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-07_0_")
space_cost.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-07_2_")

percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]


lines = ['solid', 'dotted', 'dashed', 'dashdot']

for i, el in enumerate(no_space):
    no_space_acc = []
    no_space_acc_std = []
    
    for percentage in percentages:
        acc = []
        for seed in range(seeds):
            with open(el+str(seed)+".json") as f:
                data = json.load(f)
                acc.append(data["acc_prune"+str(percentage)])
        no_space_acc.append(np.mean(acc))
        no_space_acc_std.append(np.std(acc))


    label = "Non-spatial"
    
    ax[i].errorbar(x=percentages, y=no_space_acc, yerr=no_space_acc_std, marker="o", markersize=2,
                color=palette[1], label=label, capsize=1, linewidth=0.7)


for i, el in enumerate(no_cost):
    no_cost_acc = []
    no_cost_acc_std = []
    for percentage in percentages:
        acc = []
        for seed in range(seeds):
            with open(el+str(seed)+".json") as f:
                data = json.load(f)
                acc.append(data["acc_prune"+str(percentage)])
        no_cost_acc.append(np.mean(acc))
        no_cost_acc_std.append(np.std(acc))

    label = "Spatial"
    ax[i].errorbar(x=percentages, y=no_cost_acc, yerr=no_cost_acc_std, marker="o", markersize=2,
                color=palette[0], label=label, capsize=1, linewidth=0.7)


for i, el in enumerate(space_cost):
    space_cost_acc = []
    space_cost_acc_std = []
    for percentage in percentages:
        acc = []
        for seed in range(seeds):
            with open(el+str(seed)+".json") as f:
                data = json.load(f)
                acc.append(data["acc_prune"+str(percentage)])
        space_cost_acc.append(np.mean(acc))
        space_cost_acc_std.append(np.std(acc))


    label = "Spatial+cost"
    ax[i].errorbar(x=percentages, y=space_cost_acc, yerr=space_cost_acc_std, marker="o", markersize=2,
                color=palette[3], label=label, capsize=1, linewidth=0.7)


handles, labels = [], []
regs = ['Weak', 'Medium', 'Strong', 'Extreme']
for i, a in enumerate(ax.flatten()):
    a.set_xlabel('(%) pruned')
    a.set_ylabel('Accuracy '+regs[i])
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)
    h,l = a.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    
    
fig.legend(
    handles[:3],
    labels[:3],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=False
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.subplots_adjust(top=0.96)

#plt.tight_layout()
plt.savefig("shd_pruned_acc.pdf") #76.19, 76.68, 79.20
