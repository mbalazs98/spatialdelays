import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import os
import re

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
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
fig = plt.figure(figsize=(3.25,1.7) , constrained_layout=True)
subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.1])


no_space_acc_dir = []
no_cost_acc_dir = []
space_cost_acc_dir = []

no_space_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_0_5e-10_0.05_1e-10_0_")
no_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_")
space_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_2_")


no_space_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_0_5e-11_0.05_1e-09_0_")
no_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_0_")
space_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_2_")

no_space_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_0_5e-11_0.05_1e-08_0_")
no_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_" )
space_cost_acc_dir.append("results_pos/acc_pos_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_")

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
                accs.append(float(data["val_acc"]))
        except:
            print("Missing file:", el+str(seed)+".json")
    no_space_acc.append(np.mean(accs))
    no_space_acc_std.append(np.std(accs))

print(len(no_space_acc))

label = "Non-spatial"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=no_space_acc, yerr=no_space_acc_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for el in no_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        with open(el+str(seed)+".json", "r") as f:
            data = json.load(f)
            accs.append(float(data["val_acc"]))
    no_cost_acc.append(np.mean(accs))
    no_cost_acc_std.append(np.std(accs))

label = "Spatial"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=no_cost_acc, yerr=no_cost_acc_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for el in space_cost_acc_dir:
    accs = []
    for seed in range(seeds):
        with open(el+str(seed)+".json", "r") as f:
            data = json.load(f)
            accs.append(float(data["val_acc"]))
    space_cost_acc.append(np.mean(accs))
    space_cost_acc_std.append(np.std(accs))


label = "Spatial+cost"
ax.errorbar(x=[1e-10, 1e-09, 1e-08], y=space_cost_acc, yerr=space_cost_acc_std, marker="o", markersize=3,
            color=palette[3], label=label, capsize=2, linewidth=0.7)


ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Regularisation')
ax.set_yticks([0.75, 0.8, 0.85])



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
plt.savefig("shd_reg_acc.pdf")
