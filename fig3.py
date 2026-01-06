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



fig = plt.figure(frameon=False)
ax = fig.add_subplot(projection='3d')
ax.set_axis_off()
#ax.set_facecolor('#F2F2F2')
#ax.set_position([0, 0,1, 1])  # [left, bottom, width, height]


'''ax1.set_xlabel("X")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Y")'''
dirs = "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_9/299-"
dirs2 = "nolimitnets/checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_9/"
pos0 =np.load(dirs + "Pos0.npy")
pos1 =np.load(dirs + "Pos1.npy")
pos = np.vstack([pos0, pos1]).T
N_input = 700
T= 1400
N_hidden = 128

rsta = np.zeros((N_input, T, N_hidden))
#counts = np.zeros(N_input)


z = np.load(f"{dirs2}hidden_spikes_ids.npz")
hidden_spikes_ids = [z[f] for f in z.files] 

z = np.load(f"{dirs2}hidden_spikes_times.npz")
hidden_spikes_times = [z[f].astype(int) for f in z.files] 

z = np.load(f"{dirs2}input_spikes_ids.npz")
input_spikes_ids = [z[f] for f in z.files] 

z = np.load(f"{dirs2}input_spikes_times.npz")
input_spikes_times = [(z[f]/1000).astype(int) for f in z.files] 

N_input, N_hidden, T = 700, 128, 1400


window_bins = 5  # e.g., 5 ms after each input spike
rsta = np.zeros((N_input, T, N_hidden))
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
        rsta[i_id, i_t : i_t + window_bins, :] += window
        

channel_preferred_pos = np.zeros((700,2, T))
for t in range(T):
    W_in = rsta[:, t, :]
    for c in range(700):
        w = W_in[c, :]
        nonzero_mask = w != 0  # only keep nonzero weights
        if np.any(nonzero_mask):
            wabs = np.abs(w[nonzero_mask])
            xp = (pos[nonzero_mask, 0] * wabs).sum() / wabs.sum()
            yp = (pos[nonzero_mask, 1] * wabs).sum() / wabs.sum()
        else:
            # fallback if all weights are zero (avoid NaNs)
            xp, yp = np.nan, np.nan
        channel_preferred_pos[c, :, t] = [xp, yp]

# Remove overall center bias
center = np.nanmean(channel_preferred_pos[:, 0, :]), np.nanmean(channel_preferred_pos[:, 1, :])
channel_preferred_pos[:, 0, :] -= center[0]
channel_preferred_pos[:, 1, :] -= center[1]

# Scatter plot (skip NaNs automatically)
for t in range(1000):
    ax.scatter(
        t * np.ones(700),
        channel_preferred_pos[:, 0, t],
        channel_preferred_pos[:, 1, t],
        c=np.arange(700),
        s=1,
        cmap='viridis',
        rasterized=True
    )



plt.tight_layout()

plt.savefig("rsta_plot.pdf")
