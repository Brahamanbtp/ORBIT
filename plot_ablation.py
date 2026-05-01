import matplotlib.pyplot as plt
import numpy as np

feature_sets = [
    "repetition",
    "entropy",
    "entropy + rle",
    "entropy + rep",
    "rep + rle",
    "ent + rep + rle\n(full)",
    "rle only"
]
ratios = [0.6392, 0.6193, 0.6110, 0.6117, 0.6118, 0.6121, 0.6102]
colors = ["#d62728" if i != 5 else "#2ca02c" for i in range(len(ratios))]

sorted_pairs = sorted(zip(ratios, feature_sets, colors), reverse=True)
ratios_s, fs_s, colors_s = zip(*sorted_pairs)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
y = np.arange(len(fs_s))
bars = ax.barh(y, ratios_s, color=colors_s, alpha=0.85, height=0.6)

for bar, val in zip(bars, ratios_s):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(fs_s, fontsize=9)
ax.set_xlabel("Compression Ratio (lower is better)", fontsize=10)
ax.set_title("Fig. 4. Feature Ablation — Mixed Corpus\n"
             "(green = full feature set, red = ablated)", fontsize=9)
ax.set_xlim(0.600, 0.648)
ax.axvline(0.6121, color="green", linestyle="--", linewidth=1.0,
           label="Full feature set (0.6121)")
ax.legend(fontsize=8)
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/figure4_ablation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("outputs/figure4_ablation.png", dpi=300, bbox_inches="tight")
print("Figure 4 saved.")