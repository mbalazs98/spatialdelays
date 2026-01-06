
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tonic.datasets import SHD
from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib.gridspec import GridSpecFromSubplotSpec




sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"sans-serif", "font.serif":"OpenSans"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")
fig = plt.figure(figsize=(6.75, 2.4), constrained_layout=True)
gs = GridSpec(1, 3, figure=fig, wspace=0.4)
gs2 = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,1], height_ratios=[1.7, 1.0, 0.05], hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0], projection="3d")
#ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


# -----------------------------
# Cone plotting helper
# -----------------------------
def plot_cone(ax, x0, y0, z0, z_max,
              n_r=80, n_theta=100,
              color=palette[8], alpha=0.2):

    r_max = z_max - z0
    if r_max <= 0:
        return

    r, theta = np.meshgrid(
        np.linspace(0, r_max, n_r),
        np.linspace(0, 2*np.pi, n_theta)
    )

    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z = z0 + r

    ax.plot_surface(
        x, y, z,
        color=color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
    )

# -----------------------------
# Neuron positions (columns)
# -----------------------------
neurons = {
    "a": (0, 0),
    #"b": (0, 5),
    "b": (10, 5),
    "c": (20, 5),
}

# -----------------------------
# Spike events (cone origins)
# -----------------------------
spikes = [
    (0, 0, 0),
    (10, 5, 10),
    (20, 5, 20),   # at neuron d
]

z_max = 20.7

# -----------------------------
# Plot
# -----------------------------


# Neuron columns (faint)
i = 0
for name, (x, y) in neurons.items():
    ax1.plot(
        [x, x],
        [y, y],
        [0, z_max],
        color="gray",
        alpha=1.0,
        linewidth=1
    )
    if i == 2:
        ax1.text(x, y, -2, name, color="black")
    else:
        ax1.text(x, y, -3.5, name, color="black")
    i+=1

# Cones
i = 0
for x0, y0, z0 in spikes:
    i+=1
    if i<=2:
        plot_cone(ax1, x0, y0, z0, z_max)
        ax1.scatter(x0, y0, z0, color=palette[8], s=50, marker="*", edgecolor="black", linewidth=0.5)
    else:
        ax1.scatter(x0, y0, z0, color=palette[3], s=50, marker="*", edgecolor="black", linewidth=0.5)
    

# -----------------------------
# Axes & view
# -----------------------------
#ax.set_xlim(-2, 22)
#ax.set_ylim(-2, 7)
def arrival_time(x, y, spike):
    x0, y0, z0 = spike
    return z0 + np.sqrt((x - x0)**2 + (y - y0)**2)
x = np.linspace(-2, 20, 300)
y = np.linspace(-2, 7, 200)
X, Y = np.meshgrid(x, y)
eps = 0.05  # intersection tolerance
T = []
for spike in spikes:
    T.append(arrival_time(X, Y, spike))
for i in range(len(spikes)-2):
    for j in range(i+1, len(spikes)-1):
        print(i,j)
        mask = np.abs(T[i] - T[j]) < eps

        ax1.plot(
            X[mask],
            Y[mask],
            T[i][mask],
            color=palette[3],
            linewidth=2,
            zorder=0
        )

ax1.set_zlim(0, z_max)


ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.set_xlabel("X", labelpad=-14)  # smaller pad → closer to axis
ax1.set_ylabel("Y", labelpad=-14)
ax1.set_zlabel("Time", labelpad=-16)
ax1.zaxis.label.set_rotation(-90)
ax1.grid(False)
'''ax1.set_xticks([-20, 0, 20])
ax1.set_yticks([-20, 0, 20])
ax1.set_zticks([0, 20])'''
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

dataset = SHD(save_to="../data", train=False)
raw_dataset = []
for i, data in enumerate(dataset):
    events, label = data

    raw_dataset.append((events, label))
    break


#gs = GridSpec(3, 1, figure=fig, height_ratios=[1.4, 1.3, 0.05], hspace=0.5)


ax_in = fig.add_subplot(gs2[0, 0])   # input spikes
ax_net = fig.add_subplot(gs2[1, 0])  # network
ax_out = fig.add_subplot(gs2[2, 0])  # output
# Example: replace with your real data
t = raw_dataset[0][0]["t"] / 1000.0
x = raw_dataset[0][0]["x"]

ax_in.scatter(t, x, c=x, cmap="viridis", s=0.05)
#ax_in.set_xlabel("T")
#ax_in.set_ylabel("N")

ax_in.spines.right.set_visible(False)
ax_in.spines.top.set_visible(False)
ax_in.grid(False)
ax_in.set_xticklabels([])
ax_in.set_yticklabels([])

N = 128
r = np.sqrt(np.random.uniform(0, 0.7, N))  # radius sqrt for uniform density
theta = np.random.uniform(0, 2*np.pi, N)
x = r * np.cos(theta)
y = r * np.sin(theta)
positions = (x, y)
ax_in.set_ylabel("Input channels", labelpad=-6) 
ax_in.set_xlabel("Time", labelpad=-6) 
ax_in.set_aspect("equal")
ax_net.scatter(positions[0], positions[1], s=0.02, color=palette[0])
ax_net.set_aspect("equal")
ax_net.axis("off")

circle = Circle((0, 0), 0.85, color="grey", fill=False, linewidth=0.4)
ax_net.add_patch(circle)

ax_out.axis("off")

ax_out.text(
    0.5, 0.3,
    "Loss ℓ",
    ha="center", va="center",
    #bbox=dict(boxstyle="round", fc="white", ec="black")
)

def connect_axes(fig, ax_from, ax_to, text=None, yshift=0.0, yshift2=0.0, xshift=0.0):
    bbox1 = ax_from.get_position()
    bbox2 = ax_to.get_position()

    start = (bbox1.x0 + bbox1.width/2 + xshift, bbox1.y0+yshift)
    end   = (bbox2.x0 + bbox2.width/2 + xshift, bbox2.y1+yshift2)
    style = ArrowStyle('Fancy', head_length=1, head_width=3, tail_width=0.5)
    arrow = FancyArrowPatch(
        start, end,
        transform=fig.transFigure,
        arrowstyle=style,
        linewidth=0.5,
        color="grey",
    )
    fig.add_artist(arrow)

    if text is not None:
        fig.text(
            (start[0] + end[0]) / 2-0.03,
            (start[1] + end[1]) / 2 + 0.,
            text,
            ha="center", va="center"
        )

connect_axes(fig, ax_in, ax_net, yshift=0.03, yshift2=0.02, xshift=-0.03)
connect_axes(fig, ax_net, ax_out, text="'zwei'", yshift=-0.05, yshift2=-0.06, xshift=-0.04)

bbox_net = ax_net.get_position()
bbox_out = ax_out.get_position()
'''(bbox_out.x0, bbox_out.y0 + bbox_out.height/2 - 0.15),
    (bbox_net.x1, bbox_net.y0 + bbox_net.height/2 - 0.15),'''
style = ArrowStyle('Fancy', head_length=1, head_width=3, tail_width=0.5)
arrow_err = FancyArrowPatch(
    
    (bbox_out.x0+bbox_out.width/2-0.02, bbox_out.y1-0.06),
    (bbox_net.x0+bbox_net.width/2-0.02, bbox_net.y0-0.05),
    transform=fig.transFigure,
    arrowstyle=style,
    linewidth=0.5,
    color=palette[3]
)

fig.add_artist(arrow_err)
fig.text(
            (bbox_out.x0+bbox_out.width/2-0.15 + bbox_net.x0+bbox_net.width/2-0.15) / 2+0.2,
            (bbox_out.y1 + bbox_net.y0) / 2 -0.05,
            "Position error",
            ha="center", va="center",
            color=palette[3]
        )


last_epoch = 175
dirs = "checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_18/"

first_XPos = np.load(f"{dirs}0-Pos0.npy")
first_YPos = np.load(f"{dirs}0-Pos1.npy")

Xposes = []
Yposes = []
for i in range(1, last_epoch):
    Xpos = np.load(f"{dirs}{i}-Pos0.npy")
    Ypos = np.load(f"{dirs}{i}-Pos1.npy")
    Xposes.append(Xpos)
    Yposes.append(Ypos)

last_XPos = np.load(f"{dirs}{last_epoch}-Pos0.npy")
last_YPos = np.load(f"{dirs}{last_epoch}-Pos1.npy")

ax3.plot(Xposes, Yposes, c=palette[0], linewidth=0.2)

ax3.scatter(last_XPos, last_YPos, c=palette[0], s=5)
ax3.spines.right.set_visible(False)
ax3.spines.top.set_visible(False)
ax3.grid(False)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("X")
ax3.set_ylabel("Y")

plt.savefig("introfig.pdf")
