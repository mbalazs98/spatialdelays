import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tonic.datasets import SHD
import random
import matplotlib.gridspec as gridspec


sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
palette = sns.color_palette("deep")


np.random.seed(3)
W_in_rand =np.random.normal(0.03, 0.02, (700,128))
pos0_rand =np.random.uniform(low=-1.0, high=1.0, size=128)
pos1_rand =np.random.uniform(low=-1.0, high=1.0, size=128)

fig, ax = plt.subplots(3, 4, figsize=(2.4,2.5))

positions = [a.get_position() for a in ax[0]]



dirs = [
    "",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/50-",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/100-",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/150-",


    "",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/50-",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/100-",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/150-",
]

maxs = -np.inf
for i, _dir in enumerate(dirs[4:]):
    if i != 0 and i != 4:
        print(i, _dir)
        XPos = np.load(f"{_dir}Pos0.npy")
        YPos = np.load(f"{_dir}Pos1.npy")
        points = np.column_stack([XPos, YPos])
        center = points.mean(axis=0)
        XPos = XPos - center[0]
        YPos = YPos - center[1]
        maxs = max(maxs, np.abs(XPos).max(), np.abs(YPos).max())


for i in range(4,8):
    if i == 0 or i == 4:
        W_in = W_in_rand
        pos0 = pos0_rand
        pos1 = pos1_rand
    else:
        W_in =np.load(dirs[i] + "Conn_Pop0_Pop1-g.npy").reshape(700,128)
        pos0 =np.load(dirs[i] + "Pos0.npy")
        pos1 =np.load(dirs[i] + "Pos1.npy")
    pos = np.vstack([pos0, pos1]).T
    center = pos.mean(axis=0)
    pos0 = pos0 - center[0]
    pos1 = pos1 - center[1]
    ax.flatten()[i-4].scatter(
        pos0,
        pos1,
        c=palette[0],
        s=1,
    )
    ax.flatten()[i-4].set_xticks([-maxs, 0, maxs])
    ax.flatten()[i-4].set_yticks([-maxs, 0, maxs])

    channel_preferred_pos = np.zeros((700,2))
    for c in range(700):
        w = W_in[c,:]
        # pick center-of-mass in space weighted by absolute weight
        wabs = np.abs(w) + 1e-12
        xp = (pos[:,0] * wabs).sum() / wabs.sum()
        yp = (pos[:,1] * wabs).sum() / wabs.sum()
        channel_preferred_pos[c] = [xp, yp]

    center = channel_preferred_pos[:,0].squeeze().mean(), channel_preferred_pos[:,1].squeeze().mean()
    channel_preferred_pos[:,0] -= center[0]
    channel_preferred_pos[:,1] -= center[1]

    ax.flatten()[i].scatter(
        channel_preferred_pos[:,0],
        channel_preferred_pos[:,1],
        c=np.arange(700),
        s=1,
        cmap='viridis'
    )
for i, a in enumerate(ax.flatten()):
    a.set_xticks([])
    a.set_yticks([])
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)
    if i == 0 or i == 4 or i == 8:
        ax.flatten()[i].set_ylabel('Y')
    if i > 7:
        ax.flatten()[i].set_xlabel('X')

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)

dirs2 = [
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_2_3/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/",


    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_2_3/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/",
    "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-08_0_0/",
]
e = [0, 50, 100, 150, 0, 50, 100, 150]
for i in range(4,8):
    print(i)
    z = np.load(f"{dirs2[i]}hidden_spikes_ids_{e[i]}.npz")
    hidden_spikes_ids = [z[f] for f in z.files] 

    z = np.load(f"{dirs2[i]}hidden_spikes_times_{e[i]}.npz")
    hidden_spikes_times = [z[f].astype(int) for f in z.files] 

    z = np.load(f"{dirs2[i]}input_spikes_ids.npz")
    input_spikes_ids = [z[f] for f in z.files] 

    z = np.load(f"{dirs2[i]}input_spikes_times.npz")
    input_spikes_times = [(z[f]/1000).astype(int) for f in z.files] 

    N_input, N_hidden, T = 700, 128, 1400


    window_bins = 5  # e.g., 5 ms after each input spike
    rsta = np.zeros((N_input, window_bins, N_hidden))
    counts = np.zeros(N_input)

    for s in range(len(input_spikes_ids)):
        # Build hidden spike train for this sample
        hidden_train = np.zeros((T, N_hidden), dtype=np.float32)
        hidden_train[hidden_spikes_times[s], hidden_spikes_ids[s]] = 1.0

        for i_id, i_t in zip(input_spikes_ids[s], input_spikes_times[s]):
            # Skip if spike too close to the end
            if i_t + window_bins >= T:
                continue
            # Collect hidden activity after this input spike
            window = hidden_train[i_t : i_t + window_bins, :]
            rsta[i_id] += window
            counts[i_id] += 1
            

    # Normalize
    rsta /= counts[:, None, None] + 1e-12
    W_in = rsta.mean(axis=1)  # shape (N_input, N_hidden)

    channel_preferred_pos = np.zeros((700,2))
    for c in range(700):
        w = W_in[c,:]
        # pick center-of-mass in space weighted by absolute weight
        wabs = np.abs(w) + 1e-12
        xp = (pos[:,0] * wabs).sum() / wabs.sum()
        yp = (pos[:,1] * wabs).sum() / wabs.sum()
        channel_preferred_pos[c] = [xp, yp]

    center = channel_preferred_pos[:,0].squeeze().mean(), channel_preferred_pos[:,1].squeeze().mean()
    channel_preferred_pos[:,0] -= center[0]
    channel_preferred_pos[:,1] -= center[1]
    #wbound, ebound = channel_preferred_pos[:,0].squeeze().min()-1, channel_preferred_pos[:,0].squeeze().max()+1
    #sbound, nbound = channel_preferred_pos[:,1].squeeze().min()-1, channel_preferred_pos[:,1].squeeze().max()+1
    ax.flatten()[i+4].scatter(
        channel_preferred_pos[:,0],
        channel_preferred_pos[:,1],
        c=np.arange(700),
        s=1,
        cmap='viridis'
    )

for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)
    a.grid(False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.savefig("prefpos.pdf")
