import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tonic.datasets import SHD
import random

sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

# Set palette
palette = sns.color_palette("deep")

dataset = SHD(save_to="../data", train=False)

# Loop through dataset
raw_dataset = []
for i, data in enumerate(dataset):
    events, label = data

    raw_dataset.append((events, label))
    break

ind = random.randrange(0,len(raw_dataset))


#fig, ax = plt.subplots(1, 3, figsize=(6.75,2.2))
fig = plt.figure(figsize=(6.75,5.3))
ax0 = fig.add_subplot(2, 2, 1)
ax0.scatter(raw_dataset[ind][0]["t"] / 1000.0, raw_dataset[ind][0]["x"], c=raw_dataset[ind][0]["x"], s=2, cmap='viridis') 
# Remove grid and labels
ax0.grid(False)  # Hide grid
ax0.set_yticks(np.linspace(0, 700, 5))  # Show neuron indices as ticks on the x-axis
ax0.set_xticks(np.linspace(0, 1000, 5))  # Show time as ticks on the y-axis
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_xlabel("Time (ms)")
ax0.set_ylabel("Neuron ID")

space_cost = "checkpoints_space_cartesian_limit_128_2_5e-12_0.3_62_1e-10_1_2/263-"

W_in =np.load(space_cost + "Conn_Pop0_Pop1-g.npy").reshape(700,128)
pos0 =np.load(space_cost + "Pos0.npy")
pos1 =np.load(space_cost + "Pos1.npy")
pos = np.vstack([pos0, pos1]).T
'''W_in = np.random.normal(0.03, 0.02, (700,128))
pos = np.column_stack(tuple(np.random.uniform(low=-1.0, high=1.0, size=128) for i in range(2)))'''
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

ax1 = fig.add_subplot(2, 2, 2)
asd = ax1.scatter(channel_preferred_pos[:,0].squeeze(), channel_preferred_pos[:,1].squeeze(), c=np.arange(700), s=3, cmap='viridis')


ax1.set_xlabel("X")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("Y")


space_cost = "checkpoints_space_cartesian_limit_128_2_5e-12_0.3_62_1e-10_1_2/"


z = np.load(f"{space_cost}hidden_spikes_ids.npz")
hidden_spikes_ids = [z[f] for f in z.files] 

z = np.load(f"{space_cost}hidden_spikes_times.npz")
hidden_spikes_times = [z[f].astype(int) for f in z.files] 

z = np.load(f"{space_cost}input_spikes_ids.npz")
input_spikes_ids = [z[f] for f in z.files] 

z = np.load(f"{space_cost}input_spikes_times.npz")
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
ax2 = fig.add_subplot(2, 2, 3)
asd = ax2.scatter(channel_preferred_pos[:,0].squeeze(), channel_preferred_pos[:,1].squeeze(), c=np.arange(700), s=3, cmap='viridis')

ax2.set_xlabel('X')

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = fig.add_subplot(2, 2, 4, projection='3d')


rsta = np.zeros((N_input, T, N_hidden))
#counts = np.zeros(N_input)

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
    W_in = rsta[:,t,:]
    for c in range(700):
        w = W_in[c,:]
        # pick center-of-mass in space weighted by absolute weight
        wabs = np.abs(w) + 1e-12
        xp = (pos[:,0] * wabs).sum() / wabs.sum()
        yp = (pos[:,1] * wabs).sum() / wabs.sum()
        channel_preferred_pos[c,:,t] = [xp, yp]

center = channel_preferred_pos[:,0,:].squeeze().mean(), channel_preferred_pos[:,1,:].squeeze().mean()
channel_preferred_pos[:,0,:] -= center[0]
channel_preferred_pos[:,1,:] -= center[1]
#wbound, ebound = channel_preferred_pos[:,0].squeeze().min()-1, channel_preferred_pos[:,0].squeeze().max()+1
#sbound, nbound = channel_preferred_pos[:,1].squeeze().min()-1, channel_preferred_pos[:,1].squeeze().max()+1
for t in range(1000):
    ax3.scatter(t*np.ones(700), channel_preferred_pos[:,0,t].squeeze(), channel_preferred_pos[:,1,t].squeeze(), c=np.arange(700), s=1, cmap='viridis', rasterized=True)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('X')
ax3.set_zlabel('Y')
ax3.set_xticks([0, 250, 500, 750, 1000])
ax3.set_yticks([])
ax3.set_zticks([])


ax3.grid(False)

plt.tight_layout()
plt.savefig("preferred_pos_combined.pdf")
