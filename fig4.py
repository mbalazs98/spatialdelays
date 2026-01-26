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


sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
seeds = 20
pattern = re.compile(r"^(\d+)-")
#fig, ax = plt.subplots(3,4,figsize=(6.75, 4),gridspec_kw={'height_ratios': [1, 0.5, 1]})
fig = plt.figure(figsize=(6.75,3.3) , constrained_layout=True)
subfigs = fig.subfigures(3, 1, height_ratios=[2.2, 1.3, 0.2])

dirs = [
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_0_9/166",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_2_8/129",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_0_18/123",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_2_15/163",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_5/299",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_9/299",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-07_2_3/299",
    "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-07_2_3/299",
]

maxs = [-np.inf, -np.inf, -np.inf, -np.inf]
for i, _dir in enumerate(dirs[:2]):
    XPos = np.load(f"{_dir}-Pos0.npy")
    YPos = np.load(f"{_dir}-Pos1.npy")
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    maxs[0] = max(maxs[0], np.abs(XPos).max(), np.abs(YPos).max())
for i, _dir in enumerate(dirs[2:4]):
    XPos = np.load(f"{_dir}-Pos0.npy")
    YPos = np.load(f"{_dir}-Pos1.npy")
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    maxs[1] = max(maxs[1], np.abs(XPos).max(), np.abs(YPos).max())
for i, _dir in enumerate(dirs[4:6]):
    XPos = np.load(f"{_dir}-Pos0.npy")
    YPos = np.load(f"{_dir}-Pos1.npy")
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    maxs[2] = max(maxs[2], np.abs(XPos).max(), np.abs(YPos).max())
for i, _dir in enumerate(dirs[7:]):
    XPos = np.load(f"{_dir}-Pos0.npy")
    YPos = np.load(f"{_dir}-Pos1.npy")
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    maxs[3] = max(maxs[3], np.abs(XPos).max(), np.abs(YPos).max())

abs_max = max(maxs)
maxs = [abs_max, abs_max, abs_max, abs_max]
print(abs_max)
ind=-1
ax = subfigs[0].subplots(1,4) 
titles = [
    r"$Weak\;\rightarrow$",
    r"$Medium\;\rightarrow$",
    r"$Strong\;\rightarrow$",
    r"$Extreme$"
]
for i, _dir in enumerate(dirs):
    
    if i % 2 == 0:
        colour = 0
        ind += 1
    else:
        colour = 3
    XPos = np.load(f"{_dir}-Pos0.npy")
    YPos = np.load(f"{_dir}-Pos1.npy")
    cdist_matrix = cdist(np.column_stack([XPos, YPos]), np.column_stack([XPos, YPos]))
    w = np.abs(np.load(f"{_dir}-Conn_Pop1_Pop1-g.npy").reshape(128,128))
    w_min, w_max = w.min(), w.max()
    w = (w - w_min) / (w_max - w_min)
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    if i != 6:
        for pre in range(128):
            for post in range(128):
                ax[ind].plot([XPos[pre], XPos[post]], [YPos[pre], YPos[post]], c=palette[colour], linewidth=0.1, alpha=w[pre,post], rasterized=True)
        ax[ind].scatter(XPos, YPos, s=1, c=palette[colour], alpha=0.4, rasterized=True)
        ax[ind].set_xlim(-maxs[ind], maxs[ind])
        ax[ind].set_ylim(-maxs[ind], maxs[ind])
        ax[ind].spines.right.set_visible(False)
        ax[ind].spines.top.set_visible(False)
        ax[ind].grid(False)
        #ax[ind].set_xlabel('X')
        if ind == 0:
            ax[ind].set_yticks([-500, 0, 500])
            ax[ind].set_yticklabels([-500, 0, 500])
            ax[ind].set_ylabel('y')
            ax[ind].set_xlabel('x')
        else:
            ax[ind].set_yticks([-500, 0, 500])
            ax[ind].set_yticklabels([])
            ax[ind].set_xlabel('x')

        ax[ind].set_title(titles[ind], fontweight='bold')


ax[0].text(-0.1, 1.1, "A", transform=ax[0].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )


plt.savefig("shd_reg.png")

no_space = []
no_cost = []
space_cost = []


no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-10_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-09_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_2_")


no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-08_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_")

space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-07_2_")


max_d = 0
for el in no_space:
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        max_d = max(max_d,np.max(d))
for el in no_cost:
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        max_d = max(max_d,np.max(d))
for el in space_cost:
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        max_d = max(max_d,np.max(d))
max_d = int(max_d)
print("maximum is:", max_d)
#max_d = 200
# Plot the histograms

ax = subfigs[1].subplots(1,4) 
for idx, el in enumerate(no_space):
    hist_sum = None
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        w = np.abs(np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy"))
        bins = np.linspace(0,max_d,max_d)
        # Create histograms
        hist, _ = np.histogram(d, bins=bins)
        weighting = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            # Get indices of distances within the current bin
            bin_mask = (d >= bins[i]) & (d < bins[i + 1])
            # Sum corresponding weights for positive and negative weights
            weighting[i] = np.sum(w[bin_mask])
        for i in range(len(bins) - 1):
            hist[i] = hist[i] * weighting[i]
        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
    # Histogram

    ax[idx].step(
        bins[:-1],
        hist_sum/20,
        #width=np.diff(bins),
        #facecolor='none',
        where='post', 
        color=palette[1],
        rasterized=True,
        #alpha=0.1,
        #align='edge'
        label="Non-spatial"
    ) 
    


for idx, el in enumerate(no_cost):
    hist_sum = None
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        w = np.abs(np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy"))
        bins = np.linspace(0,max_d,max_d)
        # Create histograms
        hist, _ = np.histogram(d, bins=bins)
        weighting = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            # Get indices of distances within the current bin
            bin_mask = (d >= bins[i]) & (d < bins[i + 1])
            # Sum corresponding weights for positive and negative weights
            weighting[i] = np.sum(w[bin_mask])
        for i in range(len(bins) - 1):
            hist[i] = hist[i] * weighting[i]
        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
    # Histogram
    ax[idx].step(
        bins[:-1],
        hist_sum/20,
        #width=np.diff(bins),
        #facecolor='none',
        where='post', 
        color=palette[0],
        rasterized=True,
        #alpha=0.1,
        #align='edge'
        label="Spatial"
    )   

for idx, el in enumerate(space_cost):
    hist_sum = None
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
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        w = np.abs(np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy"))
        bins = np.linspace(0,max_d,max_d)
        # Create histograms
        hist, _ = np.histogram(d, bins=bins)
        weighting = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            # Get indices of distances within the current bin
            bin_mask = (d >= bins[i]) & (d < bins[i + 1])
            # Sum corresponding weights for positive and negative weights
            weighting[i] = np.sum(w[bin_mask])
        for i in range(len(bins) - 1):
            hist[i] = hist[i] * weighting[i]
        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
    # Histogram
    ax[idx].step(
        bins[:-1],
        hist_sum/20,
        #width=np.diff(bins),
        #facecolor='none',
        where='post', 
        color=palette[3],
        rasterized=True,
        #alpha=0.1,
        #align='edge'
        label="Spatial+cost"
    )
    

for i, a in enumerate(ax):  
    a.set_yscale("log")
    a.set_ylim(0,50000000)
    a.set_xlabel('Delay')
    if i>0:
        a.set_yticklabels([])
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)

ax[0].text(-0.1, 1.1, "B", transform=ax[0].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )

ax[0].set_ylabel('Weighted count')


handles, labels = ax[0].get_legend_handles_labels()

subfigs[-1].legend(
    handles, labels,
    loc="center",
    ncol=3,
    frameon=False
)

plt.savefig("shd_reg.pdf", dpi=600)
