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


no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-10_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-10_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_5e-10_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_5e-10_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_5e-10_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-09_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-09_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_5e-09_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_5e-09_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_5e-09_2_")

no_space.append("checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_1e-08_0_")
no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_")

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


def efficiency_wei(Gw):
    '''
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
        (all weights in W must be between 0 and 1)
        
    Returns
    -------
    Eglob : float
        global efficiency
    Notes
    -----
       The  efficiency is computed using an auxiliary connection-length
    matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
    intuitive interpretation, as higher connection weights intuitively
    correspond to shorter lengths.
       The weighted local efficiency broadly parallels the weighted
    clustering coefficient of Onnela et al. (2005) and distinguishes the
    influence of different paths based on connection weights of the
    corresponding neighbors to the node in question. In other words, a path
    between two neighbors with strong connections to the node in question
    contributes more to the local efficiency than a path between two weakly
    connected neighbors. Note that this weighted variant of the local
    efficiency is hence not a strict generalization of the binary variant.

    Algorithm:  Dijkstra's algorithm
    '''

    def distance_inv_wei(G):
        n = len(G)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = G.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)

    e = distance_inv_wei(Gl)
    E = np.sum(e) / (n * n - n)
    return E

no_space_smws = []
no_space_smws_std = []
no_cost_smws = []
no_cost_smws_std = []
space_cost_smws = []
space_cost_smws_std = []
    

for el in space_cost:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        pthnull = 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for i in range(5):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            pthnull += efficiency_wei(A.reshape(128, 128))/5
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        pth = efficiency_wei(A)
        smws.append(np.divide(pth,pthnull))
    space_cost_smws.append(np.mean(smws))
    space_cost_smws_std.append(np.std(smws))
print(f'Final pathlength of cost network: {space_cost_smws[-1]}\n')
    
for el in no_space:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        pthnull = 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for i in range(5):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            pthnull += efficiency_wei(A.reshape(128, 128))/5
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        pth = efficiency_wei(A)
        smws.append(np.divide(pth,pthnull))
    no_space_smws.append(np.mean(smws))
    no_space_smws_std.append(np.std(smws))
print(f'Final small worldness of no space network: {no_space_smws[-1]}\n')


for el in no_cost:
    smws = []
    for seed in range(seeds):
        last_epoch = 0
        pthnull = 0
        for filename in os.listdir(el+str(seed)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        for i in range(5):
            a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy")
            A = np.abs(a)/np.max(np.abs(a))
            np.random.shuffle(A)
            # Take null model
            pthnull += efficiency_wei(A.reshape(128, 128))/5
        
        a = np.load(el+str(seed)+"/"+str(last_epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        A = np.abs(a)/np.max(np.abs(a))
        pth = efficiency_wei(A)
        smws.append(np.divide(pth,pthnull))
    no_cost_smws.append(np.mean(smws))
    no_cost_smws_std.append(np.std(smws))
print(f'Final smallworldness of no cost network: {no_cost_smws[-1]}\n')



ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_space_smws, yerr=no_space_smws_std,  marker='o', markersize=4,
            color=palette[1], label="Non-spatial", capsize=3, linewidth=0.7)

ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=no_cost_smws, yerr=no_cost_smws_std,  marker='o', markersize=4,
            color=palette[0], label="Spatial", capsize=3, linewidth=0.7)


ax.errorbar(x=[1e-10, 5e-10, 1e-09, 5e-09, 1e-08], y=space_cost_smws, yerr=space_cost_smws_std,  marker='o', markersize=4,
            color=palette[3], label="Spatial+cost", capsize=3, linewidth=0.7)
ax.set_xscale("log")


handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)

ax.set_ylabel('Path length')
ax.set_xlabel('Regularisation')
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

plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.subplots_adjust(top=0.96)
plt.savefig("weighted_path_length.pdf") #76.19, 76.68, 79.20
