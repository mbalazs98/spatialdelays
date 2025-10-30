import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm

sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
num_epochs = 65
seeds = 20
steps = 5

fig, ax = plt.subplots(2,3,figsize=(6.75, 4.5))
 
dirs = [
    "checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14",
    "checkpoints_space_cartesian_limit_128_2_5e-12_0.3_62_1e-10_1_2"
]

for i, _dir in enumerate(dirs):
    XPos = np.load(f"{_dir}/{250}-Pos0.npy")
    YPos = np.load(f"{_dir}/{250}-Pos1.npy")
    w = np.abs(np.load(f"{_dir}/{250}-Conn_Pop1_Pop1-g.npy").reshape(128,128))
    w_min, w_max = w.min(), w.max()
    w = (w - w_min) / (w_max - w_min)
    points = np.column_stack([XPos, YPos])
    center = points.mean(axis=0)
    XPos = XPos - center[0]
    YPos = YPos - center[1]
    for pre in range(128):
        for post in range(128):
            ax[0,i].plot([XPos[pre], XPos[post]], [YPos[pre], YPos[post]], c='black', linewidth=0.1, alpha=w[pre,post])
    ax[0,i].scatter(XPos, YPos, s=3, c=palette[0])

    ax[0,i].set_xlim(-32, 32)
    ax[0,i].set_ylim(-32, 32)
    ax[0,i].spines.right.set_visible(False)
    ax[0,i].spines.top.set_visible(False)
    ax[0,i].grid(False)
    ax[0,i].set_xlabel('X')
    ax[0,i].set_xticks([-30, 0, 30])
    ax[0,i].set_xticklabels([-30, 0, 30])
    ax[0,i].set_yticks([-30, 0, 30])
    ax[0,i].set_yticklabels([-30, 0, 30])



no_space = "checkpoints_space_cartesian_limit_128_0_5e-11_0.01_62_1e-09_0_"
no_cost = "checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_"
space_cost = "checkpoints_space_cartesian_limit_128_2_5e-12_0.3_62_1e-10_1_"
no_space_mods = []
no_space_mods_std = []
no_cost_mods = []
no_cost_mods_std = []
space_cost_mods = []
space_cost_mods_std = []

for epoch in range(0, num_epochs,steps):
    mods = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_space+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    no_space_mods.append(np.mean(mods))
    no_space_mods_std.append(np.std(mods))
print(f'Final modularity of no space network: {no_space_mods[-1]}\n')


label = "Non-spatial"
ax[1,0].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_space_mods, yerr=no_space_mods_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    mods = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    no_cost_mods.append(np.mean(mods))
    no_cost_mods_std.append(np.std(mods))
print(f'Final modularity of no cost network: {no_cost_mods[-1]}\n')

label = "Spatial"
ax[1,0].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_cost_mods, yerr=no_cost_mods_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    mods = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(space_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        c = np.abs(a)
        g = nx.from_numpy_array(c, create_using=nx.DiGraph)
        communities = nx.community.louvain_communities(g)
        q_stat = nx.community.modularity(g, communities)
        mods.append(q_stat)
    space_cost_mods.append(np.mean(mods))
    space_cost_mods_std.append(np.std(mods))
print(f'Final modularity of space cost network: {space_cost_mods[-1]}\n')

label = "Spatial+cost"
ax[1,0].errorbar(x=[e for e in range(0,num_epochs,steps)], y=space_cost_mods, yerr=space_cost_mods_std, marker="o", markersize=3,
            color=palette[4], label=label, capsize=2, linewidth=0.7)





ax[1,0].set_xlabel('Epochs')
ax[1,0].set_ylabel('Q metric')





def spectral_entropy(W):
    eigvals = np.linalg.eigvals(W)
    abs_eigvals = np.abs(eigvals)
    max_eigvals = np.max(eigvals)
    total = np.sum(abs_eigvals)
    if total == 0:
        return 0.0
    P = abs_eigvals / total
    P_nonzero = P[P > 0]
    H_spec = - np.sum(P_nonzero * np.log2(P_nonzero))
    return H_spec, max_eigvals




def shannon_entropy_weights(W):
    W = np.abs(W)
    row_sums = W.sum(axis=1, keepdims=True)
    # avoid division by zero
    row_sums[row_sums == 0] = 1
    P = W / row_sums
    P[P == 0] = 1  # avoid log2(0); log2(1)=0 so it doesn’t change result
    H_rows = -np.sum(P * np.log2(P), axis=1)
    return np.mean(H_rows)


def shannon_entropy_comms(W):
    W = np.abs(W)
    N = W.shape[0]
    s = np.diag(np.sum(W, axis=1))
    s_inv_sqrt = np.zeros_like(s)
    nonzero_mask = np.diag(s) > 0
    s_inv_sqrt[nonzero_mask, nonzero_mask] = 1.0 / np.sqrt(np.diag(s)[nonzero_mask])
    adj = s_inv_sqrt @ W @ s_inv_sqrt
    ncabs = expm(adj)
    row_sums = ncabs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = ncabs / row_sums
    P_safe = np.where(P > 0, P, 1)
    H_rows = -np.sum(P_safe * np.log2(P_safe), axis=1)
    se_ncabs = np.mean(H_rows)
    return se_ncabs

no_space_ent = []
no_space_ent_std = []
no_cost_ent = []
no_cost_ent_std = []
space_cost_ent = []
space_cost_ent_std = []
for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_space+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_weights(a))
    no_space_ent.append(np.mean(ent))
    no_space_ent_std.append(np.std(ent))
print(f'Final shannon of no space network: {no_space_ent[-1]}\n')


label = "Non-spatial"
ax[1,1].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_space_ent, yerr=no_space_ent_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_weights(a))
    no_cost_ent.append(np.mean(ent))
    no_cost_ent_std.append(np.std(ent))
print(f'Final shannon of no cost network: {no_cost_ent[-1]}\n')

label = "Spatial"
ax[1,1].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_cost_ent, yerr=no_cost_ent_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(space_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_weights(a))
    space_cost_ent.append(np.mean(ent))
    space_cost_ent_std.append(np.std(ent))
print(f'Final shannon of space cost network: {space_cost_ent[-1]}\n')

label = "Spatial+cost"
ax[1,1].errorbar(x=[e for e in range(0,num_epochs,steps)], y=space_cost_ent, yerr=space_cost_ent_std, marker="o", markersize=3,
            color=palette[4], label=label, capsize=2, linewidth=0.7)



ax[1,1].set_xlabel('Epochs')
ax[1,1].set_ylabel('Shannon entropy')



no_space_ent = []
no_space_ent_std = []
no_cost_ent = []
no_cost_ent_std = []
space_cost_ent = []
space_cost_ent_std = []
for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_space+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    no_space_ent.append(np.mean(ent))
    no_space_ent_std.append(np.std(ent))
print(f'Final shannon of no space network: {no_space_ent[-1]}\n')

#fig, ax = plt.subplots()
label = "Non-spatial"
ax[0,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_space_ent, yerr=no_space_ent_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    no_cost_ent.append(np.mean(ent))
    no_cost_ent_std.append(np.std(ent))
print(f'Final shannon of no cost network: {no_cost_ent[-1]}\n')

label = "Spatial"
ax[0,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_cost_ent, yerr=no_cost_ent_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(space_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(shannon_entropy_comms(a))
    space_cost_ent.append(np.mean(ent))
    space_cost_ent_std.append(np.std(ent))
print(f'Final shannon of space cost network: {space_cost_ent[-1]}\n')

label = "Spatial+cost"
ax[0,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=space_cost_ent, yerr=space_cost_ent_std, marker="o", markersize=3,
            color=palette[4], label=label, capsize=2, linewidth=0.7)



ax[0,2].set_xlabel('Epochs')
ax[0,2].set_ylabel('Communicibality entropy')
ax[0,2].set_yticks([4, 4.5, 5])


no_space_ent = []
no_space_ent_std = []
no_cost_ent = []
no_cost_ent_std = []
space_cost_ent = []
space_cost_ent_std = []
for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_space+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(spectral_entropy(a)[0])
    no_space_ent.append(np.mean(ent))
    no_space_ent_std.append(np.std(ent))
print(f'Final spectral of no space network: {no_space_ent[-1]}\n')

label = "Non-spatial"
ax[1,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_space_ent, yerr=no_space_ent_std, marker="o", markersize=3,
            color=palette[1], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(no_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(spectral_entropy(a)[0])
    no_cost_ent.append(np.mean(ent))
    no_cost_ent_std.append(np.std(ent))
print(f'Final spectral of no cost network: {no_cost_ent[-1]}\n')

label = "Spatial"
ax[1,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=no_cost_ent, yerr=no_cost_ent_std, marker="o", markersize=3,
            color=palette[0], label=label, capsize=2, linewidth=0.7)

for epoch in range(0, num_epochs,steps):
    ent = []
    print(epoch)
    for seed in range(seeds):
        a = np.load(space_cost+str(seed)+"/"+str(epoch)+"-Conn_Pop1_Pop1-g.npy").reshape(128, 128)
        ent.append(spectral_entropy(a)[0])
    space_cost_ent.append(np.mean(ent))
    space_cost_ent_std.append(np.std(ent))
print(f'Final spectral of space cost network: {space_cost_ent[-1]}\n')

label = "Spatial+cost"
ax[1,2].errorbar(x=[e for e in range(0,num_epochs,steps)], y=space_cost_ent, yerr=space_cost_ent_std, marker="o", markersize=3,
            color=palette[4], label=label, capsize=2, linewidth=0.7)



ax[1,2].set_xlabel('Epochs')
ax[1,2].set_ylabel('Spectral entropy')
handles, labels = [], []
for i, a in enumerate(ax[1:,:].flatten()):
    h, l = a.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    break

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig("shd_epoch.pdf")
