import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/block_size_sweep.json") as f:
    data = json.load(f)

block_sizes = [d["block_size"] for d in data]
ratios = [d["compression_ratio"] for d in data]
throughputs = [d["throughput_mbps"] for d in data]
conv_blocks = [d["regret_convergence_block"] for d in data]

x = np.arange(len(block_sizes))
labels = ["1 KB", "4 KB", "16 KB", "64 KB"]

fig, ax1 = plt.subplots(figsize=(6, 3.5))

color_ratio = "#1f77b4"
color_tp = "#d62728"

# Compression ratio bars
bars = ax1.bar(x - 0.2, ratios, width=0.35, color=color_ratio,
               alpha=0.8, label="Compression Ratio")
ax1.set_ylabel("Compression Ratio (lower is better)", color=color_ratio, fontsize=9)
ax1.tick_params(axis="y", labelcolor=color_ratio)
ax1.set_ylim(0.4, 1.05)

# Throughput on secondary axis
ax2 = ax1.twinx()
ax2.bar(x + 0.2, throughputs, width=0.35, color=color_tp,
        alpha=0.8, label="Throughput (MB/s)")
ax2.set_ylabel("Throughput (MB/s)", color=color_tp, fontsize=9)
ax2.tick_params(axis="y", labelcolor=color_tp)
ax2.set_ylim(0, 18)

# Convergence block annotations
for i, cb in enumerate(conv_blocks):
    ax1.text(x[i] - 0.2, ratios[i] + 0.01, f"C={cb}", fontsize=7,
             ha="center", color="navy")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_xlabel("Block Size", fontsize=10)

# Highlight default
ax1.axvline(x=1, color="gray", linestyle=":", linewidth=1.0)
ax1.text(1, 0.42, "default\n(4 KB)", fontsize=7, ha="center", color="gray")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

plt.title("Fig. 2. Block Size Sensitivity: Ratio, Throughput, and Convergence",
          fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figure2_block_size.pdf", dpi=300, bbox_inches="tight")
plt.savefig("outputs/figure2_block_size.png", dpi=300, bbox_inches="tight")
print("Figure 2 saved.")