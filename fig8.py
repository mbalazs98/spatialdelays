import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import os
import re

import json

sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
seeds = 20
pattern = re.compile(r"^(\d+)-")
#fig, ax = plt.subplots(3,4,figsize=(6.75, 4),gridspec_kw={'height_ratios': [1, 0.5, 1]})
fig = plt.figure(figsize=(3.25,3) , constrained_layout=True)
subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 0.1])


no_space_acc_dir = []
no_cost_acc_dir = []
space_cost_acc_dir = []

no_space_acc_dir.append("results_pos/prune_nolimit_128_0_5e-13_0.05_1e-10_0_")
no_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-10_0_")
space_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-10_2_")

no_space_acc_dir.append("results_pos/prune_nolimit_128_0_5e-13_0.05_5e-10_0_")
no_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_5e-10_0_")
space_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_5e-10_2_")


no_space_acc_dir.append("results_pos/prune_nolimit_128_0_5e-13_0.05_1e-09_0_")
no_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-09_0_")
space_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-09_2_")

no_space_acc_dir.append("results_pos/prune_nolimit_128_0_5e-13_0.05_5e-09_0_")
no_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_5e-09_0_")
space_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_5e-09_2_")

no_space_acc_dir.append("results_pos/prune_nolimit_128_0_5e-13_0.05_1e-08_0_")
no_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-08_0_" )
space_cost_acc_dir.append("results_pos/prune_nolimit_128_2_5e-13_0.05_1e-08_2_")


ax = subfigs[0].subplots(1,1) 
no_cost_acc = []
no_cost_acc_std = []
space_cost_acc = []
space_cost_acc_std = []
no_space_acc = []
no_space_acc_std = []
for el in no_space_acc_dir:
    accs = []
    for seed in range(seeds):
        try:
            with open(el+str(seed)+".json", "r") as f:
                data = json.load(f)
                accs.append(float(data["acc_prune0"]))
                if float(data["acc_prune0"]) <= float(data["acc_prune100"]):
                    print("Warning: pruned accuracy less than unpruned:", el+str(seed)+".json")
                else:
                    accs.append(float(data["acc_prune0"]))
        except:
            print("Missing file:", el+str(seed)+".json")
    no_space_acc.append(np.mean(accs))
    no_space_acc_std.append(np.std(accs))


label = "Non-spatial"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_space_acc, yerr=no_space_acc_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for el in no_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        with open(el+str(seed)+".json", "r") as f:
            data = json.load(f)
            
            if float(data["acc_prune0"]) <= float(data["acc_prune100"]):
                    print("Warning: pruned accuracy less than unpruned:", el+str(seed)+".json")
            else:
                accs.append(float(data["acc_prune0"]))
    no_cost_acc.append(np.mean(accs))
    no_cost_acc_std.append(np.std(accs))

label = "Spatial"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_cost_acc, yerr=no_cost_acc_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for el in space_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        with open(el+str(seed)+".json", "r") as f:
            data = json.load(f)
            accs.append(float(data["acc_prune0"]))
            if float(data["acc_prune0"]) <= float(data["acc_prune100"]):
                print("Warning: pruned accuracy less than unpruned:", el+str(seed)+".json")
            else:
                accs.append(float(data["acc_prune0"]))
    space_cost_acc.append(np.mean(accs))
    space_cost_acc_std.append(np.std(accs))


label = "Spatial+cost"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=space_cost_acc, yerr=space_cost_acc_std, marker="o", markersize=3,
            color=palette[3], label=label, capsize=2, linewidth=0.7)


ax.set_ylabel('Accuracy (%)')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)


ax.set_xscale("log")
ax.set_xticks([])
ax.set_yticks([0.8, 0.85])


ax.text(-0.1, 1.1, "A", transform=ax.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )

ax = subfigs[1].subplots(1,1) 
no_cost_acc = []
no_cost_acc_std = []
space_cost_acc = []
space_cost_acc_std = []
no_space_acc = []
no_space_acc_std = []
for el in no_space_acc_dir:
    accs = []
    for seed in range(seeds):
        try:
            with open(el+str(seed)+".json", "r") as f:
                data = json.load(f)
                try:
                    accs.append(float(data["prune"]))
                except KeyError:
                    print("Key 'prune' not found in:", el+str(seed)+".json")
        except FileNotFoundError:
            print("File not found:", el+str(seed)+".json")
    no_space_acc.append(np.mean(accs))
    no_space_acc_std.append(np.std(accs))


label = "Non-spatial"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_space_acc, yerr=no_space_acc_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for el in no_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        try:
            with open(el+str(seed)+".json", "r") as f:
                data = json.load(f)
                try:
                    accs.append(float(data["prune"]))
                except KeyError:
                    print("Key 'prune' not found in:", el+str(seed)+".json")
        except:
            print("File not found:", el+str(seed)+".json")
    no_cost_acc.append(np.mean(accs))
    no_cost_acc_std.append(np.std(accs))

label = "Spatial"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_cost_acc, yerr=no_cost_acc_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for el in space_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        try:
            with open(el+str(seed)+".json", "r") as f:
                data = json.load(f)
                try:
                    accs.append(float(data["prune"]))
                except KeyError:
                    print("Key 'prune' not found in:", el+str(seed)+".json")
        except:
            print("File not found:", el+str(seed)+".json")
    space_cost_acc.append(np.mean(accs))
    space_cost_acc_std.append(np.std(accs))


label = "Spatial+cost"
ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=space_cost_acc, yerr=space_cost_acc_std, marker="o", markersize=3,
            color=palette[3], label=label, capsize=2, linewidth=0.7)


ax.set_ylabel('Sparsity (%)')
ax.set_xlabel('Regularisation')



handles, labels = ax.get_legend_handles_labels()

subfigs[-1].legend(
    handles, labels,
    loc="center",
    ncol=3,
    frameon =False
)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)


ax.set_xscale("log")

titles = [
    "   Weak" + r"$\rightarrow$",
    "   Medium" + r"$\rightarrow$",
    "Strong"
]
ax.set_xticks([1e-10, 1e-09, 1e-08])
ax.set_xticklabels(titles)
#ax.set_yticks([35, 70])
#ax.set_yticklabels(["0.35", "0.70"])

ax.text(-0.1, 1.1, "B", transform=ax.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )


plt.savefig("shd_reg_acc.png")
