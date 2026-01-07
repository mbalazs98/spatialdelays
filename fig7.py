import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
sns.set_palette("deep")


# Folder with results

twod = ["results_pos/acc_pos_cartesian_32_2_1.0_5e-12_0.01_",
"results_pos/acc_pos_cartesian_64_2_1.0_5e-12_0.03_",
"results_pos/acc_pos_cartesian_128_2_1.0_5e-13_0.01_"
]

threed = ["results_pos/acc_pos_cartesian_32_3_1.0_5e-12_0.01_",
"results_pos/acc_pos_cartesian_64_3_1.0_5e-12_0.01_",
"results_pos/acc_pos_cartesian_128_3_1.0_5e-13_0.03_"
]

fourd = ["results_pos/acc_pos_cartesian_32_4_1.0_5e-13_0.01_",
"results_pos/acc_pos_cartesian_64_4_1.0_5e-13_0.03_",
"results_pos/acc_pos_cartesian_128_4_1.0_5e-13_0.003_"
]

nodelay = ["results_pos/acc_pos_cartesian_32_0_1.0_5e-12_0.0_",
    "results_pos/acc_pos_cartesian_64_0_1.0_5e-11_0.0_",
    "results_pos/acc_pos_cartesian_128_0_1.0_5e-09_0.0_"]

nospace = ["results_pos/acc_nospace_32_1.0_5e-11_0.1_",
    "results_pos/acc_nospace_64_1.0_5e-11_0.1_",
    "results_pos/acc_nospace_128_1.0_5e-12_0.5_"
]

axon = ["results_pos/acc_nospace_axon_32_1.0_5e-13_1.0_",
"results_pos/acc_nospace_axon_64_1.0_5e-12_0.05_",
"results_pos/acc_nospace_axon_128_1.0_5e-13_0.1_"]

architectures = [threed, twod, nospace, nodelay, fourd, axon]
architecturenames = ["3D", "2D", "non-spatial", "non-spatial no delay", "4D", "axon"]
palette = sns.color_palette("deep")

fig, ax = plt.subplots(figsize=(3.25,2.5))

sizes = [32, 64, 128]
colors = {'3D': palette[0], '2D': palette[0], 'non-spatial': palette[1], 'non-spatial no delay': palette[2], '4D': palette[0], 'axon': palette[4]}
markers = {'3D': 's', '2D': 'o', 'non-spatial': 'o', 'non-spatial no delay': 'o', '4D': '^', 'axon': 'o'}
for archname, arch in zip(architecturenames, architectures):
    val_means = []
    val_stds = []
    size_vals = []
    for i in range(len(sizes)):
        vals = []
        for seed in range(20):
            try:
                with open(arch[i]+str(seed)+".json") as f:
                    data = json.load(f)
                vals.append(float(data['val_acc']))
            except:
                print(f"File {arch[i]+str(seed)+'.json'} not found")
        label = f"{archname}"
        if archname == "3D":
            size = sizes[i] * 3 + (sizes[i] **2)
        elif archname == "2D":
            size = sizes[i] * 3 + (sizes[i] **2)
        elif archname == "non-spatial":
            size = (sizes[i] **2) * 2
        elif archname == "non-spatial no delay":
            size = (sizes[i] **2)
        elif archname == "4D":
            size = sizes[i] * 4 + (sizes[i] **2)
        elif archname == "axon":
            size = sizes[i] + (sizes[i] **2)
        val_means.append(np.mean(vals))
        val_stds.append(np.std(vals))
        size_vals.append(size)

    ax.errorbar(x=size_vals, y=val_means, yerr=val_stds,  marker=markers.get(archname, None), markersize=3,
                color=colors.get(archname, None), label=label, capsize=2, linewidth=0.7,linestyle="None")

ax.set_xlabel('Number of trainable parameters')
ax.set_ylabel('Test accuracy (%)')
handles, labels = [], []
h, l = ax.get_legend_handles_labels()

handles.extend(h)
labels.extend(l)
print(labels)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
plt.xscale("log")
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.show()
plt.savefig("shd_acc_space.pdf")
