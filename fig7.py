import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
sns.set_palette("deep")

# Folder with results
pattern = re.compile(r"^(\d+)-")
results_dir = 'results_pos'

data = []
palette = sns.color_palette("deep")
nospace_full = "acc_pos_cartesian_nolimit_dynamic_128_0_5e-13_0.05_0.0_0_"
space_full = "acc_pos_cartesian_nolimit_dynamic_128_2_5e-13_0.05_0.0_0_"
nospace_prune = "prunedist_nolimit_128_0_5e-13_0.05_0.0_0_"
space_prune = "prunedist_nolimit_128_2_5e-13_0.05_0.0_0_"
checkpoint_nospace = "checkpoints_space_cartesian_nolimit_dynamic_128_0_5e-13_0.05_0.0_0_"
checkpoint_space = "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_0.0_0_"
nospace_accs = np.zeros((7, 20))
space_accs = np.zeros((7, 20))
for seed in range(20):
    with open(os.path.join(results_dir, f"{nospace_full}{seed}.json")) as f:
            res = json.load(f)
            nospace_accs[0, seed] = float(res["val_acc"])
    with open(os.path.join(results_dir, f"{nospace_prune}{seed}.json")) as f:
            res = json.load(f)
            nospace_accs[1, seed] = float(res["acc95"])
            nospace_accs[2, seed] = float(res["acc90"])
            nospace_accs[3, seed] = float(res["acc85"])
            nospace_accs[4, seed] = float(res["acc80"])
            nospace_accs[5, seed] = float(res["acc75"])
            nospace_accs[6, seed] = float(res["acc70"])
    with open(os.path.join(results_dir, f"{space_full}{seed}.json")) as f:
            res = json.load(f)
            space_accs[0, seed] = float(res["val_acc"])
    with open(os.path.join(results_dir, f"{space_prune}{seed}.json")) as f:
            res = json.load(f)
            space_accs[1, seed] = float(res["acc95"])
            space_accs[2, seed] = float(res["acc90"])
            space_accs[3, seed] = float(res["acc85"])
            space_accs[4, seed] = float(res["acc80"])
            space_accs[5, seed] = float(res["acc75"])
            space_accs[6, seed] = float(res["acc70"])



fig, ax = plt.subplots(2, 2 ,figsize=(6.75, 3.5))

        
ax[0,0].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=np.mean(nospace_accs, 1), yerr=np.std(nospace_accs, 1),  marker='s', markersize=4,
            color=palette[1], label="Non-spatial", capsize=3, linewidth=0.7)

ax[0,0].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=np.mean(space_accs, 1), yerr=np.std(space_accs, 1),  marker='s', markersize=4,
            color=palette[0], label="Spatial", capsize=3, linewidth=0.7)

ax[0,0].set_xlabel('(%) of longest connections pruned')
ax[0,0].set_ylabel('Test accuracy (%)')


max_d = 0
for i in range(0,19):
    last_epoch = 0
    for filename in os.listdir(checkpoint_nospace+ str(i)):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Convert to integer
            if num > last_epoch:
                last_epoch = num
    if last_epoch < 299:
        last_epoch -= 16
    d = np.load(checkpoint_nospace+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
    max_d = max(max_d,np.max(d))
    last_epoch = 0
    for filename in os.listdir(checkpoint_space+ str(i)):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Convert to integer
            if num > last_epoch:
                last_epoch = num
    if last_epoch < 299:
        last_epoch -= 16
    d = np.load(checkpoint_space+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
    max_d = max(max_d,np.max(d))

max_d = 200
# Plot the histograms

hist_sum = None
for seed in range(0,19):
    last_epoch = 0
    for filename in os.listdir(checkpoint_nospace+ str(i)):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Convert to integer
            if num > last_epoch:
                last_epoch = num
    if last_epoch < 299:
        last_epoch -= 16
    d = np.load(checkpoint_nospace+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
    bins = np.linspace(0,int(max_d))



    # Create histograms
    hist, _ = np.histogram(d, bins=bins)
    if hist_sum is None:
        hist_sum = hist
    else:
        hist_sum += hist

# Histogram
for i in range(len(bins) - 1):
    ax0 = ax[0,1].bar(
        bins[i],
        hist_sum[i]/20,
        width=bins[i + 1] - bins[i],
        color=palette[1],
        alpha=0.75
    )

hist_sum = None
for seed in range(0,19):
    last_epoch = 0
    for filename in os.listdir(checkpoint_space+ str(seed)):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # Convert to integer
            if num > last_epoch:
                last_epoch = num
    if last_epoch < 299:
        last_epoch -= 16
    d = np.load(checkpoint_space+ str(seed)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy")
    bins = np.linspace(0,int(max_d))



    # Create histograms
    hist, _ = np.histogram(d, bins=bins)
    if hist_sum is None:
        hist_sum = hist
    else:
        hist_sum += hist

# Histogram
for i in range(len(bins) - 1):
    ax1 = ax[0,1].bar(
        bins[i],
        hist_sum[i]/20,
        width=bins[i + 1] - bins[i],
        color=palette[0],
        alpha=0.75
    )





ax[0,1].set_ylabel('Counts')
ax[0,1].set_xlabel('Delay')


def modularity_und(binary_weight_matrix, gamma=1):
    """
    Manual modularity calculation using Louvain algorithm
    Returns: (communities, modularity_value)
    """
    A = np.array(binary_weight_matrix, dtype=np.float64)
    A = (A + A.T) / 2  # Make symmetric
    A = (A > 0).astype(np.float64)  # Ensure binary

    n = A.shape[0]
    m = np.sum(A) / 2  # Total edges

    if m == 0:
        return np.arange(n), 0.0

    degrees = np.sum(A, axis=1)
    communities = np.arange(n)  # Start with each node in own community

    # Louvain algorithm
    improved = True
    while improved:
        improved = False
        for node in np.random.permutation(n):
            current_comm = communities[node]
            best_comm = current_comm
            best_gain = 0.0

            # Check neighboring communities
            neighbors = np.where(A[node] > 0)[0]
            neighbor_comms = set(communities[neighbors])

            for new_comm in neighbor_comms:
                if new_comm == current_comm:
                    continue

                # Calculate modularity gain
                gain = 0.0
                for j in range(n):
                    if j != node:
                        if communities[j] == new_comm:
                            gain += 2 * (A[node, j] - gamma * degrees[node] * degrees[j] / (2 * m))
                        elif communities[j] == current_comm:
                            gain -= 2 * (A[node, j] - gamma * degrees[node] * degrees[j] / (2 * m))

                if gain > best_gain:
                    best_gain = gain
                    best_comm = new_comm

            if best_gain > 0:
                communities[node] = best_comm
                improved = True

    # Calculate final modularity
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                Q += A[i, j] - gamma * (degrees[i] * degrees[j]) / (2 * m)
    Q = Q / (2 * m)

    return communities, Q


percentages = [95, 90, 85, 80, 75, 70]

mean_2d, std_2d = [], []
mean_2d.append(0)
std_2d.append(0)
for percentage in percentages:
    q_stats = []
    for i in range(20):
        last_epoch = 0
        for filename in os.listdir(checkpoint_space+ str(i)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        d = np.load(checkpoint_space+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        ci, q_stat = modularity_und(d < np.percentile(np.abs(d), percentage), gamma=1)
        q_stats.append(q_stat)
    mean_2d.append(np.mean(q_stats))
    std_2d.append(np.std(q_stats))

mean_nospace, std_nospace = [], []
mean_nospace.append(0)
std_nospace.append(0)
for percentage in percentages:
    q_stats = []
    for i in range(20):
        last_epoch = 0
        for filename in os.listdir(checkpoint_nospace+ str(i)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        d = np.load(checkpoint_nospace+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        ci, q_stat = modularity_und(d < np.percentile(np.abs(d), percentage), gamma=1)
        q_stats.append(q_stat)
    mean_nospace.append(np.mean(q_stats))
    std_nospace.append(np.std(q_stats))


ax[1,0].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=mean_nospace, yerr=std_nospace,  marker='s', markersize=4,
                    color=palette[1], label="Non-spatial", capsize=3, linewidth=0.7)

ax[1,0].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=mean_2d, yerr=std_2d,  marker='s', markersize=4,
                    color=palette[0], label="Spatial", capsize=3, linewidth=0.7)

ax[1,0].set_xlabel('(%) of longest connections pruned')
ax[1,0].set_ylabel('Modularity')



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


def clustering_coef_bu(G):
    '''
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    n = len(G)
    C = np.zeros((n,))

    for u in range(n):
        V, = np.where(G[u, :])
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)]
            C[u] = np.sum(S) / (k * k - k)

    return C

def efficiency_bin(G, local=False):
    '''
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.

    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected connection matrix
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.

    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 np.ndarray
        local efficiency, only if local=True
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
    if local:
        E = np.zeros((n,))  # local efficiency

        for u in range(n):
            # V,=np.where(G[u,:])			#neighbors
            # k=len(V)					#degree
            # if k>=2:					#degree must be at least 2
            #	e=distance_inv(G[V].T[V])
            #	E[u]=np.sum(e)/(k*k-k)	#local efficiency computation

            # find pairs of neighbors
            V, = np.where(np.logical_or(G[u, :], G[u, :].T))
            # inverse distance matrix
            e = distance_inv(G[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = e + e.T

            # symmetrized adjacency vector
            sa = G[u, V] + G[V, u].T
            numer = np.sum(np.outer(sa.T, sa) * se) / 2
            if numer != 0:
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                E[u] = numer / denom  # local efficiency

    else:
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
    cluperm = np.zeros((nperm,1))
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
        cluperm[perm] = np.mean(clustering_coef_bu(Aperm))
        #pthperm[perm] = efficiency_bin(Aperm)
    # Take the average of the nulls
    clunull = np.mean(cluperm)
    pthnull = np.mean(pthperm)
    for i in range(20):
        last_epoch = 0
        for filename in os.listdir(checkpoint_space+ str(i)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        d = np.load(checkpoint_space+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        A = d < np.percentile(np.abs(d), percentage)
        # Compute the small worldness
        clu = np.mean(clustering_coef_bu(A))
        #pth = efficiency_bin(A)
        #smws.append(np.divide(np.divide(clu,clunull),np.divide(pthnull,pth)))
        smws.append(np.divide(clu,clunull))

    space_mean.append(np.mean(smws))
    space_std.append(np.std(smws))

nospace_mean, nospace_std  = [], []
nospace_mean.append(1)
nospace_std.append(0)
for percentage in percentages:
    
    # Run nperm null models
    nperm = 1000
    cluperm = np.zeros((nperm,1))
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
        cluperm[perm] = np.mean(clustering_coef_bu(Aperm))
        #pthperm[perm] = efficiency_bin(Aperm)
    # Take the average of the nulls
    clunull = np.mean(cluperm)
    pthnull = np.mean(pthperm)
    for i in range(20):
        last_epoch = 0
        for filename in os.listdir(checkpoint_nospace+ str(i)):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))  # Convert to integer
                if num > last_epoch:
                    last_epoch = num
        if last_epoch < 299:
            last_epoch -= 16
        d = np.load(checkpoint_nospace+ str(i)+ "/"+str(last_epoch)+"-Conn_Pop1_Pop1-d.npy").reshape(128,128)
        #print(np.percentile(np.abs(d), 95))
        A = d < np.percentile(np.abs(d), percentage)
        # Compute the small worldness
        clu = np.mean(clustering_coef_bu(A))
        #pth = efficiency_bin(A)
        #smws.append(np.divide(np.divide(clu,clunull),np.divide(pthnull,pth)))
        smws.append(np.divide(clu,clunull))

    nospace_mean.append(np.mean(smws))
    nospace_std.append(np.std(smws))


ax[1,1].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=nospace_mean, yerr=nospace_std,  marker='s', markersize=4,
            color=palette[1], label="Non-spatial", capsize=3, linewidth=0.7)

ax[1,1].errorbar(x=[0, 5, 10, 15, 20, 25, 30], y=space_mean, yerr=space_std,  marker='s', markersize=4,
            color=palette[0], label="Spatial", capsize=3, linewidth=0.7)
ax[1,1].set_xlabel('(%) of longest connections pruned')
ax[1,1].set_ylabel('Clustering')
handles, labels = [], []
for i, a in enumerate(ax[1:,:].flatten()):
    h, l = a.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    break
for a in ax.flatten():
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.subplots_adjust(top=0.93)
ax[0,0].text(-0.1, 1.1, "A", transform=ax[0,0].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
ax[0,1].text(-0.1, 1.1, "B", transform=ax[0,1].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold')
ax[1,0].text(-0.1, 1.1, "C", transform=ax[1,0].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold')
ax[1,1].text(-0.1, 1.1, "D", transform=ax[1,1].transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold')
#plt.tight_layout()
plt.show()
plt.savefig("shd_prune_all.pdf")
