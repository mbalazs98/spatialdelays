import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tonic.datasets import SHD
import random
import matplotlib.gridspec as gridspec


sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":7, "xtick.labelsize":7, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})
# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
palette = sns.color_palette("deep")


fig, ax = plt.subplots(2, 1, figsize=(1.7,1.3), gridspec_kw={'height_ratios':[50, 1]})
#fig.suptitle(r"$Training\;\longrightarrow$",)


spatial = [0.7007212668657303, 0.6554879501461983, 0.6796232640743256, 0.7150551825761795, 0.6912143185734749]
spatial_std = [0.07197070004629212, 0.05830402303507012, 0.08772186818573299, 0.06797978008077776, 0.0540967889298935]
spatial_cost = [0.7054103091359138, 0.6645516932010651, 0.6700219377875328, 0.7350346788764, 0.7364875942468643]
spatial_cost_std = [0.06533690999255047, 0.051360170145055825, 0.056631963113808505, 0.042435296643249985, 0.06007930690088353]

spatial = [0.6386759951710701, 0.5655000373721123, 0.5337595865130424, 0.7773673757910728, 0.7292693704366684]
spatial_std = [0.07557132778585604, 0.0727792399555431, 0.060908262626759996, 0.0765402506006128, 0.0544895580384884]
spatial_cost = [0.6022074490785598, 0.582078218460083, 0.5689807772636414, 0.7833317518234253, 0.9054001972079277]
spatial_cost_std = [0.07447559146677715, 0.09728878957698563, 0.061433898320741746, 0.05220834087846558, 0.04459526797853547]

ax[0].errorbar(
    [1e-10, 5e-10, 1e-9, 5e-9, 1e-8],
    spatial,
    yerr=spatial_std,
    color=palette[0],
    label='Spatial',
)

ax[0].errorbar(
    [1e-10, 5e-10, 1e-9, 5e-9, 1e-8],
    spatial_cost,
    yerr=spatial_cost_std,
    color=palette[3],
    label='Spatial\n+cost',
)
ax[0].set_xscale("log")
ax[0].set_yticks([0, 1])
ax[0].set_xticks([1e-10, 1e-9, 1e-8])
ax[0].set_xticklabels(["Weak","Medium","Strong"])
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[0].grid(False)
ax[0].set_ylabel(r'R$^2$ score')

#ax.set_xlabel('Regularisation')

'''ax.legend(loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=1)'''
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
    columnspacing=0.5,
    fontsize=7,
)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[1].spines.left.set_visible(False)
ax[1].spines.bottom.set_visible(False)
ax[1].grid(False)
ax[1].set_yticks([])
ax[1].set_xticks([])

plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig("reg.pdf")
