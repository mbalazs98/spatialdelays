
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tonic.datasets import SHD
from matplotlib.patches import Circle, FancyArrowPatch, ArrowStyle
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import matplotlib as mpl





sns.set(context="paper", rc={"font.size":8, "axes.labelsize":8, "axes.titlesize": 9,
                                 "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8})
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times"})
# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

palette = sns.color_palette("deep")

fig = plt.figure(figsize=(6.75, 2.1), constrained_layout=True)
gs = GridSpec(1, 3, figure=fig, wspace=0)
gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1], height_ratios=[1, 1])
gs3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[0,0],wspace=0.01)

ax1 = fig.add_subplot(gs[0, 0], projection="3d")
#ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


# -----------------------------
# Cone plotting helper
# -----------------------------

def draw_axis_arrows(ax, origin=(0, 0, 0),
                     space_len=5, time_len=6,
                     lw=0.3):
    x0, y0, z0 = origin

    # --- Time arrow (single direction) ---
    ax.quiver(
        x0, y0, z0,
        0, 0, time_len,
        color="grey",
        linewidth=lw,
        arrow_length_ratio=0.15
    )
    ax.text(x0, y0, z0 + time_len + 0.5, "Time", ha="center")

    # --- Space arrows (multiple directions) ---
    directions = [
        (space_len, 0, 0),
        (-space_len, 0, 0),
        (0, space_len, 0),
        (0, -space_len, 0),
    ]

    for dx, dy, dz in directions:
        ax.quiver(
            x0, y0, z0,
            dx, dy, dz,
            color="grey",
            linewidth=lw,
            arrow_length_ratio=0.15
        )

    ax.text(x0+3, y0 - space_len-0.5, z0, "Space", ha="left")


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
    "a": (0-3, 0-2),
    "b": (10-3, 5-2),
    "c": (2.37, 17.27),
}

# -----------------------------
# Spike events (cone origins)
# -----------------------------
spikes = [
    (0-3, 0-2, 0),
    (10-3, 5-2, 5),
    (2.37, 17.27, 20.7),   # at neuron c
]

z_max = 20.7

# -----------------------------
# Plot
# -----------------------------

draw_axis_arrows(
    ax1,
    origin=(0, 0, 0),
    space_len=30,
    time_len=30
)

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
        ax1.scatter(x0, y0, z0, color=palette[8], s=50, marker="*", edgecolor="black", linewidth=0.5)
    


ax1.set_zlim(0, z_max)

z_obs = z_max
def draw_wavefront_circle(ax, x0, y0, z0, z_obs,
                          n_theta=400,
                          color="C3",
                          lw=2):
    r = z_obs - z0
    if r <= 0:
        return

    theta = np.linspace(0, 2*np.pi, n_theta)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z = np.full_like(theta, z_obs)

    ax.plot(
        x, y, z,
        color=color,
        linewidth=lw,
        zorder=5
    )
    return x,y, r

centres = []
radiuses = []
for x0, y0, z0 in spikes[:2]:
    x, y, r = draw_wavefront_circle(
        ax1,
        x0, y0, z0,
        z_obs,
        color="black",
        lw=0.5
    )
    centres.append((x0, y0))
    radiuses.append(r)




def circle_outline_intersections(c0, r0, c1, r1):
    x0, y0 = c0
    x1, y1 = c1

    dx = x1 - x0
    dy = y1 - y0
    d = np.hypot(dx, dy)

    # No outline intersections
    if d > r0 + r1:
        return []          # separate circles
    if d < abs(r0 - r1):
        return []          # one inside another
    if d == 0 and r0 == r1:
        return []          # coincident outlines (infinite points)

    # Geometry
    a = (r0**2 - r1**2 + d**2) / (2*d)
    h2 = r0**2 - a**2
    if h2 < 0:
        return []          # numerical safety

    h = np.sqrt(h2)

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry =  dx * (h / d)

    if h == 0:
        # Tangential touch → exactly one outline point
        return [(xm, ym)]

    return [
        (xm + rx, ym + ry),
        (xm - rx, ym - ry),
    ]


pts = circle_outline_intersections(centres[0], radiuses[0], centres[1], radiuses[1])
for (x, y) in pts:
    print(x,y, z_obs)
    ax1.scatter(
        x, y, z_obs,
        color=palette[3],
        s=2.0,
        zorder=10
    )

z_obs = z_max
xs1, ys1, xs2, ys2, zs, = [], [], [], [], []
for curr_zobs in range(5, (int(z_max)+1)):
    for i in range(len(spikes)-2):
        for j in range(i + 1, len(spikes)):
            x0, y0, z0 = spikes[i]
            x1, y1, z1 = spikes[j]

            r0 = curr_zobs - z0
            r1 = curr_zobs - z1

            pts = circle_outline_intersections(
                (x0, y0), r0,
                (x1, y1), r1
            )
            if pts:
                xs1.append(pts[0][0])
                xs2.append(pts[1][0])
                ys1.append(pts[0][1])
                ys2.append(pts[1][1])
                zs.append(curr_zobs)

xs1.insert(0, xs2[0])
xs2.insert(0, xs1[0])
zs.insert(0, zs[0])
ys1.insert(0, ys2[0])
ys2.insert(0, ys1[0])
ax1.plot3D(xs1, ys1, zs, color=palette[3], linewidth=1.0, zorder=10)
ax1.plot3D(xs2, ys2, zs, color=palette[3], linewidth=1.0, zorder=10)
print(xs1[-1], ys1[-1])

ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax1.grid(False)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.w_xaxis.line.set_color((0, 0, 0, 0))
ax1.w_yaxis.line.set_color((0, 0, 0, 0))
ax1.w_zaxis.line.set_color((0, 0, 0, 0))
# hide pane edges and grid lines
star_proxy = Line2D([], [], marker='*', linestyle='None', markersize=7,
                    markerfacecolor=palette[8], markeredgecolor='black', linewidth=0.1)
neuron_pos = Line2D([], [], marker='.', linestyle='None', markersize=7,
                    markerfacecolor='grey', markeredgecolor='grey', linewidth=0.1)
ax1.legend(handles=[star_proxy, neuron_pos], labels=["Spike", "Position"], loc="center left", frameon=False)

dataset = SHD(save_to="../data", train=False)
raw_dataset = []
for i, data in enumerate(dataset):
    events, label = data
    raw_dataset.append((events, label))
    break




ax_in = fig.add_subplot(gs3[0, 0])   # input spikes
ax_net = fig.add_subplot(gs3[0, 1])  # network
ax_out = fig.add_subplot(gs2[1, 0])  # output
t = raw_dataset[0][0]["t"] / 1000.0
x = raw_dataset[0][0]["x"]

ax_in.scatter(t, x, c=x, cmap="viridis", s=0.05,marker= "_")


ax_in.spines.right.set_visible(False)
ax_in.spines.top.set_visible(False)
ax_in.grid(False)
ax_in.set_xticklabels([])
ax_in.set_yticklabels([])

N = 32
r = np.sqrt(np.random.uniform(0, 0.9, N))  # radius sqrt for uniform density
theta = np.random.uniform(0, 2*np.pi, N)
x = r * np.cos(theta)
y = r * np.sin(theta)
positions = (x, y)
ax_in.set_ylabel("Channels", labelpad=-6) 
ax_in.set_xlabel("Time", labelpad=-6) 

y_vals = positions[1]
threshold = np.percentile(y_vals, 2)  # bottom half
candidates = np.where(y_vals < threshold)[0]
if len(candidates) >= 2:
    xs_candidates = positions[0][candidates]
    left_idx = candidates[np.argmin(xs_candidates)]
    right_idx = candidates[np.argmax(xs_candidates)]
    idxs = np.array([left_idx, right_idx])
else:
    # fallback: pick two points with smallest y (deterministic)
    idxs = np.argsort(y_vals)[:2]
ax_net.scatter(positions[0], positions[1], s=5, color=palette[0])
highlight_radius = 0.06

for idx in idxs:
    circ = Circle(
        (positions[0][idx], positions[1][idx]),
        highlight_radius,
        facecolor="none",
        edgecolor=palette[3],
        linewidth=0.6,
        zorder=10
    )
    ax_net.add_patch(circ)

ax_net.set_aspect("equal")


ax_net.spines.right.set_visible(False)
ax_net.spines.top.set_visible(False)
ax_net.grid(False)
ax_net.set_xticklabels([])
ax_net.set_yticklabels([])
ax_net.set_xlabel("x", labelpad=-6)
ax_net.set_ylabel("y", labelpad=-6)


bbox_net = ax_net.get_position()


# --- Define Neuron Positions ---
# Chosen coordinates to ensure diff(x) and diff(y) have visibly different lengths
# Neuron i (the focus)
xi, yi = 2.0, 3.0
# Neuron j (the neighbor)
xj, yj = 2.8, 3.3

# --- Calculate Geometry and Vectors ---
# Vector from j to i (direction of repulsive force)
vec_ji_x = xi - xj
vec_ji_y = yi - yj
d_ij = np.sqrt(vec_ji_x**2 + vec_ji_y**2)

# Define a scaling factor for the gradient magnitude for visualization purposes
# This represents (dL/ddij + dL/ddji) * learning_rate
grad_scalar_magnitude = .7

# Calculate the actual update vector components based on the formula
# Formula component: (xi - xj) / dij
update_x = (vec_ji_x / d_ij) * grad_scalar_magnitude
update_y = -(vec_ji_y / d_ij) * grad_scalar_magnitude + 0.1

# --- Drawing Style Parameters ---
neuron_radius = 0.12
neuron_color = palette[0] # standard matplotlib blue
grad_color = palette[3]    # standard matplotlib red
conn_color = 'gray'

# --- 1. Draw Neurons i and j ---
# Neuron i

ax_out.scatter(
    [xi], [yi],
    s=400,
    color=neuron_color,
    zorder=10
)

ax_out.text(xi, yi, r"$\mathregular{x_i, y_i}$", ha='center', va='center', color='white', zorder=11)

# Neuron j
ax_out.scatter(
    [xj], [yj],
    s=400,
    color=neuron_color,
    zorder=10
)

ax_out.text(xj, yj, r"$\mathregular{x_j, y_j}$", ha='center', va='center', color='white', zorder=11)

# --- 2. Draw Bidirectional Connections ---
# Connection i -> j
conn_ij = FancyArrowPatch((xi, yi), (xj, yj),
                          connectionstyle="arc3,rad=0.0",
                          arrowstyle='->', color=conn_color, lw=1.5, ls='-')
ax_out.add_patch(conn_ij)

# Label distance
ax_out.text((xi+xj)/2, (yi+yj)/2 - 0.15, r"$\mathregular{d_{ij}=d_{ji}}$", ha='center', color=conn_color)

# target neurons in ax_out
targets = [(xi, yi), (xj, yj)]

style = ArrowStyle(
    "Fancy",
    head_length=1.2,
    head_width=3,
    tail_width=0.6
)




# --- 3. Draw Gradient Update Arrows from Neuron i ---

# A) The Sum Arrow (The actual spatial movement vector)
# Drawn first so it's slightly behind the components
sum_arrow = FancyArrowPatch((xi, yi), (xi + update_x, yi + update_y),
                            arrowstyle='Simple, head_width=8, head_length=10',
                            color=grad_color, lw=2, alpha=0.5, zorder=5)
ax_out.add_patch(sum_arrow)



# B) The X-axis Component Arrow
# Note: We draw from xi to xi+update_x, keeping y constant at yi
arrow_x = FancyArrowPatch((xi, yi), (xi + update_x, yi),
                          arrowstyle='->, head_width=5, head_length=5',
                          color=grad_color, lw=2, zorder=6)
ax_out.add_patch(arrow_x)

# Label for X component
label_x = r"$\mathregular{\frac{x_i - x_j}{d_{ij}} \left( \frac{\partial L}{\partial d_{ij}} + \frac{\partial L}{\partial d_{ji}} \right)}$"
# Position text below the X arrow
ax_out.text(xi + update_x/2, yi - 0.2, label_x,
        color=grad_color, ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=2))


# C) The Y-axis Component Arrow
# Note: We draw from yi to yi+update_y, keeping x constant at xi
arrow_y = FancyArrowPatch((xi, yi), (xi, yi + update_y),
                          arrowstyle='->, head_width=5, head_length=5',
                          color=grad_color, lw=2, zorder=6)
ax_out.add_patch(arrow_y)

# Label for Y component
label_y = r"$\mathregular{\frac{y_i - y_j}{d_{ij}} \left( \frac{\partial L}{\partial d_{ij}} + \frac{\partial L}{\partial d_{ji}} \right)}$"
# Position text to the left of the Y arrow
ax_out.text(xi+0.25, yi + update_y/2+0.25, label_y,
        color=grad_color, ha='right', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=2))

# Set plot limits with padding
ax_out.set_xlim(xi - 0.7, xj + 0.2)
ax_out.set_ylim(yi - .2, yj + .2)

ax_out.spines.right.set_visible(False)
ax_out.spines.top.set_visible(False)
ax_out.spines.left.set_visible(False)
ax_out.spines.bottom.set_visible(False)
ax_out.grid(False)
ax_out.set_xticklabels([])
ax_out.set_yticklabels([])

from matplotlib.patches import ConnectionPatch

for src_idx, (tx, ty) in zip(idxs, targets):
    sx, sy = positions[0][src_idx], positions[1][src_idx]

    con = ConnectionPatch(
        xyA=(sx, sy), xyB=(tx, ty+0.13),
        coordsA="data", coordsB="data",
        axesA=ax_net, axesB=ax_out,
        arrowstyle=style,
        mutation_scale=3,    # tune head size
        linewidth=1,
        color="grey",
        alpha=0.2,
        zorder=2
    )
    fig.add_artist(con)

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

custom_map = sns.blend_palette(
    [(0, 1, 1), palette[0]],
    as_cmap=True
)

import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=1, vmax=len(Xposes)-1)


for i in range(len(Xposes)):
    ax3.scatter(Xposes[i], Yposes[i], cmap=custom_map, c=i*np.ones(128), norm=norm, s=0.5)



ax3.scatter(last_XPos, last_YPos, c=palette[0], s=5)
# Add a colorbar to indicate epoch progression for the hull/trajectory points
# use the same colormap and normalization used when plotting the intermediate poses
try:
    sm = mpl.cm.ScalarMappable(cmap=custom_map, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\leftarrow$Training", rotation=270, labelpad=10)
    cbar.set_ticks([])
except Exception:
    pass
ax3.spines.right.set_visible(False)
ax3.spines.top.set_visible(False)
ax3.grid(False)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("x", labelpad=-6)
ax3.set_ylabel("y", labelpad=-6)

ax1.text2D(-0.1, 1.1, "A", transform=ax1.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
ax_in.text(-0.1, 1.2, "B", transform=ax_in.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
ax_net.text(-0.1, 1.2, "C", transform=ax_net.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
ax_out.text(-0.1, 1.1, "D", transform=ax_out.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
ax3.text(-0.1, 1.1, "E", transform=ax3.transAxes,
            fontsize=8, va='top', ha='left',fontweight='bold' )
plt.savefig("introfig.pdf", dpi=300)
