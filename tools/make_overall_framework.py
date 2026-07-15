import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PNG = "fig_overall_framework_refined_final.png"
OUT_PDF = "fig_overall_framework_refined_final.pdf"

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")


def add_box(x, y, w, h, text, facecolor, edgecolor="#1f4e5f"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        wrap=True
    )
    return box


def add_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.6,
        color="#9b2f2f"
    )
    ax.add_patch(arrow)


# =========================
# COLORS
# =========================
flood_col = "#eaf4ff"
sim_col = "#e9f7ef"
ml_col = "#fff2cc"
control_col = "#fce4d6"
eval_col = "#eadcf8"

# =========================
# TITLE
# =========================
ax.text(
    7.5,
    7.55,
    "Overall Architecture of the Proposed Flood-Aware Transformer-PPO Framework",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold"
)

# =========================
# TOP ROW
# =========================
add_box(
    0.6, 5.6, 2.5, 1.4,
    "Flood Severity\n\n0.00 m  Baseline\n0.12 m  Mild\n0.28 m  Moderate\n0.55 m  Severe",
    flood_col
)

add_box(
    4.0, 5.75, 2.3, 1.1,
    "SUMO + TraCI\nTraffic Simulation",
    sim_col
)

add_box(
    7.2, 5.75, 2.5, 1.1,
    "Traffic State Vector\nQueue • Wait • Speed\nFlood Depth • Phase",
    sim_col
)

add_box(
    10.7, 5.75, 2.7, 1.1,
    "Lightweight Transformer\nTemporal State Encoder",
    ml_col
)

# =========================
# MIDDLE ROW
# =========================
add_box(
    10.7, 3.55, 2.7, 1.15,
    "PPO Actor--Critic Agent\nPolicy + Value Learning",
    ml_col
)

add_box(
    7.2, 3.55, 2.5, 1.15,
    "Adaptive Signal Actions\nPhase Selection\nGreen-Time Allocation",
    control_col
)

add_box(
    3.9, 3.55, 2.5, 1.15,
    "Updated Network State\nVehicle Arrivals\nQueue Length\nAverage Waiting",
    control_col
)

# =========================
# BOTTOM ROW
# =========================
add_box(
    3.9, 1.25, 2.5, 1.1,
    "Performance Metrics\nThroughput • Waiting\nService Rate • Queue",
    eval_col
)

add_box(
    7.2, 1.25, 2.5, 1.1,
    "Antifragility Index\nAI = Throughput Ratio\n− Delay Penalty",
    eval_col
)

add_box(
    10.7, 1.25, 2.7, 1.1,
    "Reward Function\n+ Throughput\n− Delay\n− Queue\n+ AI Bonus",
    eval_col
)

# =========================
# MAIN FLOW ARROWS
# =========================
add_arrow(3.1, 6.3, 4.0, 6.3)
add_arrow(6.3, 6.3, 7.2, 6.3)
add_arrow(9.7, 6.3, 10.7, 6.3)

add_arrow(12.05, 5.75, 12.05, 4.7)
add_arrow(10.7, 4.1, 9.7, 4.1)
add_arrow(7.2, 4.1, 6.4, 4.1)

add_arrow(5.15, 3.55, 5.15, 2.35)
add_arrow(6.4, 1.8, 7.2, 1.8)
add_arrow(9.7, 1.8, 10.7, 1.8)

# Reward feedback to PPO
add_arrow(12.05, 2.35, 12.05, 3.55)

# =========================
# NOTE
# =========================
ax.text(
    7.5,
    0.45,
    "Closed-loop evaluation: network response updates the performance metrics, Antifragility Index, and reward signal used for PPO policy improvement.",
    ha="center",
    fontsize=11
)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")