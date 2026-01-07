import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
sns.set_palette("deep")

# Folder with results
results_dir = 'results_pos'

data = []
palette = sns.color_palette("deep")
nospace_full = "acc_nospace_128_1.0_5e-12_0.5_"
space_full = "acc_pos_cartesian_128_2_1.0_5e-13_0.01_"
nospace_prune = "prune_nospace_128_1.0_5e-12_0.5_"
space_prune = "prune_pos_cartesian128_2_1.0_5e-13_0.01_"
checkpoint_nospace = "checkpoints_shd_nospace128_1.0_5e-12_0.5_"
checkpoint_space = "checkpoints_space_cartesian128_2_1.0_5e-13_0.01_"


fig, ax = plt.subplots(1, 1 ,figsize=(3.25, 2.5))



percentages = [95, 90, 85, 80, 75, 70]



def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def efficiency_bin(G):
    '''
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    '''
    def distance_inv(g):
        D = np.eye(len(g))
        n = 1
        nPATH = g.copy()
        L = (nPATH != 0)

        while np.any(L):
            D += n * L
            n += 1
            nPATH = np.dot(nPATH, g)
            L = (nPATH != 0) * (D == 0)
        D[np.logical_not(D)] = np.inf
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    G = binarize(G)
    n = len(G)  # number of nodes
    e = distance_inv(G)
    E = np.sum(e) / (n * n - n)  # global efficiency
    return E
    
        

percentages = [95, 90, 85, 80, 75, 70]
space_mean, space_std  = [], []
space_mean.append(1)
space_std.append(0)
for percentage in percentages:
    
    # Run nperm null models
    nperm = 1000
    pthperm = np.zeros((nperm,1))
    smws = []
    for perm in range(nperm):
        #Wperm = np.random.rand(100,100)
        Wperm = np.random.uniform(0, 1, size=(128,128))
        # Make it into a matrix
        Wperm = np.matrix(Wperm)
        # Make symmetrical
        #Wperm = Wperm+Wperm.T
        #Wperm = np.divide(Wperm,2)
        # Binarise
        #threshold, upper, lower = .7,1,0
        threshold, upper, lower = 1-(percentage/100),1,0
        Aperm = np.where(Wperm>threshold,upper,lower)
        # Take null model
        pthperm[perm] = efficiency_bin(Aperm)
    # Take the average of the nulls
    pthnull = np.mean(pthperm)
    for i in range(20):
        d = np.load(checkpoint_space+ str(i)+ "/best-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        A = d < np.percentile(np.abs(d), percentage)
        # Compute the small worldness
        pth = efficiency_bin(A)
        smws.append(np.divide(pth,pthnull))

    space_mean.append(np.mean(smws))
    space_std.append(np.std(smws))

nospace_mean, nospace_std  = [], []
nospace_mean.append(1)
nospace_std.append(0)
for percentage in percentages:
    
    # Run nperm null models
    nperm = 1000
    pthperm = np.zeros((nperm,1))
    smws = []
    for perm in range(nperm):
        #Wperm = np.random.rand(100,100)
        Wperm = np.random.uniform(0, 1, size=(128,128))
        # Make it into a matrix
        Wperm = np.matrix(Wperm)
        # Make symmetrical
        #Wperm = Wperm+Wperm.T
        #Wperm = np.divide(Wperm,2)
        # Binarise
        #threshold, upper, lower = .7,1,0
        threshold, upper, lower = 1-(percentage/100),1,0
        Aperm = np.where(Wperm>threshold,upper,lower)
        # Take null model
        pthperm[perm] = efficiency_bin(Aperm)
    # Take the average of the nulls
    pthnull = np.mean(pthperm)
    for i in range(20):
        d = np.load(checkpoint_nospace+ str(i)+ "/best-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        A = d < np.percentile(np.abs(d), percentage)
        # Compute the small worldness
        pth = efficiency_bin(A)
        smws.append(np.divide(pth,pthnull))

    nospace_mean.append(np.mean(smws))
    nospace_std.append(np.std(smws))


ax.errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=nospace_mean, yerr=nospace_std,  marker='s', markersize=4,
            color=palette[1], label="Non-spatial", capsize=3, linewidth=0.7)

ax.errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=space_mean, yerr=space_std,  marker='s', markersize=4,
            color=palette[0], label="Spatial", capsize=3, linewidth=0.7)
ax.set_xlabel('(%) of longest connections pruned')
ax.set_ylabel('Path length')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.subplots_adjust(top=0.93)
#plt.tight_layout()
plt.savefig("binary_path_length.pdf")
