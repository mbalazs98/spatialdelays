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
fig = plt.figure(figsize=(6.75,1.7) , constrained_layout=True)
subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.1])




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

ax = subfigs[0].subplots(1,3) 

no_space_mods = []
no_space_mods_std = []
no_cost_mods = []
no_cost_mods_std = []
space_cost_mods = []
space_cost_mods_std = []

for el in no_space:
    mods = []
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
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    no_space_mods.append(np.mean(mods))
    no_space_mods_std.append(np.std(mods))
print(f'Final modularity of no space network: {no_space_mods[-1]}\n')


label = "Non-spatial"
ax[0].errorbar(x=[1e-10, 1e-09, 1e-08], y=no_space_mods, yerr=no_space_mods_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)


for el in no_cost:
    mods = []
    
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
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    no_cost_mods.append(np.mean(mods))
    no_cost_mods_std.append(np.std(mods))
print(f'Final modularity of no cost network: {no_cost_mods[-1]}\n')

label = "Spatial"
ax[0].errorbar(x=[1e-10, 1e-09, 1e-08], y=no_cost_mods, yerr=no_cost_mods_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for el in space_cost:
    mods = []
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
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    space_cost_mods.append(np.mean(mods))
    space_cost_mods_std.append(np.std(mods))
print(f'Final modularity of space cost network: {space_cost_mods[-1]}\n')

label = "Spatial+cost"
ax[0].errorbar(x=[1e-10, 1e-09, 1e-08], y=space_cost_mods, yerr=space_cost_mods_std, marker="o", markersize=3,
            color=palette[3], label=label, capsize=2, linewidth=0.7)


ax[0].set_ylabel('Modularity')


ax[0].text(-0.1, 1.1, "A", transform=ax[0].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )


def cuberoot(x):
    '''
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    '''
    return np.sign(x) * np.abs(x)**(1 / 3)

def invert(W, copy=True):
    '''
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W

def clustering_coef_wd(W):
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector

    Notes
    -----
    Methodological note (also see clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version

    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make C=0
    # number of all possible 3 cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3  # clustering coefficient
    return C

no_space_smws = []
no_space_smws_std = []
no_cost_smws = []
no_cost_smws_std = []
space_cost_smws = []
space_cost_smws_std = []

#nperm = 1000

max_val = 0
for el in zip(no_space, no_cost, space_cost):
    for e in el:
        for seed in range(seeds):
            last_epoch = 0
            for filename in os.listdir(e+str(seed)):
                match = pattern.match(filename)
                if match:
                    num = int(match.group(1))  # Convert to integer
                    if num > last_epoch:
                        last_epoch = num
            if last_epoch < 299:
                last_epoch -= 16
            a = np.load(e+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            max_val = max(max_val, np.max(np.abs(a)))
maxs = [max_val, max_val, max_val, max_val]
print(maxs)


for el in space_cost:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        clunull, pthnull = 0, 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for _ in range(50):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            clunull += np.mean(clustering_coef_wd(A.reshape(128, 128)))/50
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        clu = np.mean(clustering_coef_wd(A))
        smws.append(np.divide(clu,clunull))
    space_cost_smws.append(np.mean(smws))
    space_cost_smws_std.append(np.std(smws))
print(f'Final smallworldness of cost network: {space_cost_smws[-1]}\n')
    
for el in no_space:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        clunull = 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for i in range(50):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            clunull += np.mean(clustering_coef_wd(A.reshape(128, 128)))/50
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        clu = np.mean(clustering_coef_wd(A))
        smws.append(np.divide(clu,clunull))
    no_space_smws.append(np.mean(smws))
    no_space_smws_std.append(np.std(smws))
    print("No space:", no_space_smws[-1])
print(f'Final small worldness of no space network: {no_space_smws[-1]}\n')


for el in no_cost:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        clunull = 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for i in range(50):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            clunull += np.mean(clustering_coef_wd(A.reshape(128, 128)))/50
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        clu = np.mean(clustering_coef_wd(A))
        smws.append(np.divide(clu,clunull))
    no_cost_smws.append(np.mean(smws))
    no_cost_smws_std.append(np.std(smws))
    print("No cost:", no_cost_smws[-1])
print(f'Final smallworldness of no cost network: {no_cost_smws[-1]}\n')

ax[1].errorbar(x=[1e-10, 1e-09, 1e-08], y=no_space_smws, yerr=no_space_smws_std, marker="o", markersize=3,
            color=palette[1], label="Non-spatial", capsize=2, linewidth=0.7)

ax[1].errorbar(x=[1e-10, 1e-09, 1e-08], y=no_cost_smws, yerr=no_cost_smws_std, marker="o", markersize=3,
            color=palette[0], label="Spatial", capsize=2, linewidth=0.7)

ax[1].errorbar(x=[1e-10, 1e-09, 1e-08], y=space_cost_smws, yerr=space_cost_smws_std, marker="o", markersize=3,
            color=palette[3], label='Spatial+cost', capsize=2, linewidth=0.7)

ax[1].set_ylabel('Clustering')

ax[1].text(-0.1, 1.1, "B", transform=ax[1].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )


'''handles, labels = ax[2,0].get_legend_handles_labels()'''

'''fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3)'''


space_wirings = []
space_wirings_stds = []
for el in space_cost:
    results = []
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
        w = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
        w = np.abs(w)
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        wiring_cost = np.sum(w * d)
        minimum_wiring = np.sum(np.sort(w) * np.flip(np.sort(d)))
        maximum_wiring = np.sum(np.sort(w) * np.sort(d))
        results.append((wiring_cost-minimum_wiring)/(maximum_wiring-minimum_wiring))
    space_wirings.append(np.mean(results))
    space_wirings_stds.append(np.std(results))

no_space_wirings = []
no_space_wirings_stds = []
for el in no_space:
    results = []
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
        w = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
        w = np.abs(w)
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        wiring_cost = np.sum(w * d)
        minimum_wiring = np.sum(np.sort(w) * np.flip(np.sort(d)))
        maximum_wiring = np.sum(np.sort(w) * np.sort(d))
        results.append((wiring_cost-minimum_wiring)/(maximum_wiring-minimum_wiring))
    no_space_wirings.append(np.mean(results))
    no_space_wirings_stds.append(np.std(results))

no_cost_wirings = []
no_cost_wirings_stds = []
for el in no_cost:
    results = []
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
        w = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
        w = np.abs(w)
        d = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
        wiring_cost = np.sum(w * d)
        minimum_wiring = np.sum(np.sort(w) * np.flip(np.sort(d)))
        maximum_wiring = np.sum(np.sort(w) * np.sort(d))
        results.append((wiring_cost-minimum_wiring)/(maximum_wiring-minimum_wiring))
    no_cost_wirings.append(np.mean(results))
    no_cost_wirings_stds.append(np.std(results))

ax[0].axhline(0.4967, color='grey', linestyle='--', linewidth=0.5)

# Add text label slightly above the line
ax[0].text(
    3e-10, 0.5167,                # x, y position in data coordinates
    "Human",
    fontsize=6,
    #verticalalignment='bottom'
)

# Add dashed horizontal line at y=2.5
ax[1].axhline(1.59, color='grey', linestyle='--', linewidth=0.5)

# Add text label slightly above the line
ax[1].text(
    3e-09, 1.64,                # x, y position in data coordinates
    "Macaque",
    fontsize=6,
    #verticalalignment='bottom'
)


ax[2].errorbar(x=[1e-10, 1e-09, 1e-08], y=[1-no_space_wiring for no_space_wiring in no_space_wirings], yerr=no_space_wirings_stds, marker="o", markersize=3,
            color=palette[1], label="Non-spatial", capsize=2, linewidth=0.7)

ax[2].errorbar(x=[1e-10, 1e-09, 1e-08], y=[1-no_cost_wiring for no_cost_wiring in no_cost_wirings], yerr=no_cost_wirings_stds, marker="o", markersize=3,
            color=palette[0], label="Spatial", capsize=2, linewidth=0.7)

ax[2].errorbar(x=[1e-10, 1e-09, 1e-08], y=[1-space_wiring for space_wiring in space_wirings], yerr=space_wirings_stds, marker="o", markersize=3,
            color=palette[3], label='Spatial+cost', capsize=2, linewidth=0.7)

ax[2].set_ylabel('Wiring efficiency')
for a in ax:
    a.set_xscale("log")

ax[2].text(-0.1, 1.1, "C", transform=ax[2].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
handles, labels = ax[0].get_legend_handles_labels()

for a in ax:
    a.set_xlabel('Regularisation')

subfigs[-1].legend(
    handles, labels,
    loc="center",
    ncol=3,
    frameon =False
)
titles = [
    "   Weak" + r"$\rightarrow$",
    "   Medium" + r"$\rightarrow$",
    "Strong"
]
for a in ax:
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)
    a.set_xticks([1e-10, 1e-09, 1e-08])
    a.set_xticklabels(titles)

plt.savefig("shd_reg2.pdf")
