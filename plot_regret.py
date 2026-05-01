import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
with open("outputs/core_mixed_corpus/regret_curve_aggregated.json") as f:
    data = json.load(f)

block_ids = [d["block_id"] for d in data]
mean_regret = [d["mean_normalized_regret"] for d in data]
std_regret = [d["std_normalized_regret"] for d in data]

mean_arr = np.array(mean_regret)
std_arr = np.array(std_regret)
x = np.array(block_ids)

fig, ax = plt.subplots(figsize=(7, 3.5))

# Mean curve
ax.plot(x, mean_arr, color="#1f77b4", linewidth=1.5, label="Mean normalized regret")

# Std band
ax.fill_between(x, mean_arr - std_arr, mean_arr + std_arr,
                alpha=0.25, color="#1f77b4", label=r"$\pm$1 std dev")

# Convergence marker
conv_block = 293
ax.axvline(x=conv_block, color="red", linestyle="--", linewidth=1.0,
           label=f"Convergence (block {conv_block})")

# Phase annotations
ax.axvspan(0, 20, alpha=0.08, color="orange")
ax.axvspan(20, conv_block, alpha=0.08, color="green")
ax.text(10, 0.19, "Burn-in", fontsize=7, ha="center", color="darkorange")
ax.text(156, 0.19, "Learning phase", fontsize=7, ha="center", color="darkgreen")
ax.text(3000, 0.002, "Converged", fontsize=7, ha="center", color="navy")

ax.set_xlabel("Block Index", fontsize=10)
ax.set_ylabel("Normalized Cumulative Regret", fontsize=10)
ax.set_title("Fig. 1. Normalized Cumulative Regret over Blocks (Mixed Corpus, 5 Runs)",
             fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, 5002)
ax.set_ylim(-0.005, 0.25)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/figure1_regret_curve.pdf", dpi=300, bbox_inches="tight")
plt.savefig("outputs/figure1_regret_curve.png", dpi=300, bbox_inches="tight")
print("Figure 1 saved.")