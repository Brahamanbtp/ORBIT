import matplotlib.pyplot as plt
import numpy as np

corpora = ["Mixed", "Text", "Binary"]
lz4_vals   = [16.0,  6.8, 26.4]
zstd_vals  = [60.0, 85.0, 30.2]
lzma_vals  = [ 6.8,  8.1,  7.2]
raw_vals   = [17.1,  0.1, 36.2]

x = np.arange(len(corpora))
width = 0.18

colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

fig, ax = plt.subplots(figsize=(6.5, 3.5))

b1 = ax.bar(x - 1.5*width, lz4_vals,  width, label="LZ4",  color=colors[0], alpha=0.85)
b2 = ax.bar(x - 0.5*width, zstd_vals, width, label="Zstd", color=colors[1], alpha=0.85)
b3 = ax.bar(x + 0.5*width, lzma_vals, width, label="LZMA", color=colors[2], alpha=0.85)
b4 = ax.bar(x + 1.5*width, raw_vals,  width, label="Raw",  color=colors[3], alpha=0.85)

# Value labels
for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        if h > 1.0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Blocks Assigned (%)", fontsize=10)
ax.set_xlabel("Corpus", fontsize=10)
ax.set_title("Fig. 3. Codec Selection Distribution per Corpus (ORBIT Policy)",
             fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(corpora, fontsize=10)
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/figure3_codec_distribution.pdf", dpi=300, bbox_inches="tight")
plt.savefig("outputs/figure3_codec_distribution.png", dpi=300, bbox_inches="tight")
print("Figure 3 saved.")