import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tonic.datasets import SHD
import random
import matplotlib.gridspec as gridspec


sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":7, "xtick.labelsize":7, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
palette = sns.color_palette("deep")


fig, ax = plt.subplots(2, 1, figsize=(1.45,1.3),height_ratios=[1, 0.1])
#fig.suptitle(r"$Training\;\longrightarrow$",)


spatial = [0.7, 0.68, 0.73]
spatial_std = [0.05, 0.07, 0.05]
spatial_cost = [0.68, 0.67, 0.91]
spatial_cost_std = [0.08, 0.06, 0.05]

ax[0].errorbar(
    [0, 1, 2],
    spatial,
    yerr=spatial_std,
    color=palette[0],
    label='Spatial',
)

ax[0].errorbar(
    [0, 1, 2],
    spatial_cost,
    yerr=spatial_cost_std,
    color=palette[3],
    label='Spatial\n+cost',
)

ax[0].set_yticks([0, 1])
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(["Weak","Medium","Strong"])
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[0].grid(False)
ax[0].set_ylabel('R2 score')
#ax.set_xlabel('Regularisation')

'''ax.legend(loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=1)'''
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].legend(
    handles, labels,
    loc="center",
    ncol=2,
    frameon=False,
    columnspacing=0.5,
)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[1].spines.left.set_visible(False)
ax[1].spines.bottom.set_visible(False)
ax[1].grid(False)
ax[1].set_yticks([])
ax[1].set_xticks([])

plt.tight_layout()

plt.savefig("reg.pdf")
