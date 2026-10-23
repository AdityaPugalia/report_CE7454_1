"""Figure: example image per class with its NPR residual (amplified), plus a 64x64 zoom."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8})
tr = sorted(Path("dev_phase/train").glob("*.jpg"))
by = {c: [p for p in tr if p.stem.endswith(f"_{c}")] for c in range(3)}
rng = np.random.default_rng(3)
picks = [by[c][rng.integers(len(by[c]))] for c in range(3)]
names = ["Real (0)", "Model A (1)", "Model B (2)"]
def npr(x):
    d = F.interpolate(x, scale_factor=0.5, mode="nearest")
    return x - F.interpolate(d, size=x.shape[-2:], mode="nearest")
fig, axes = plt.subplots(3, 3, figsize=(5.4, 5.4))
for i, p in enumerate(picks):
    im = np.array(Image.open(p).convert("RGB"))
    x = torch.from_numpy(im).permute(2,0,1).float().div(255).sub(0.5).div(0.5)[None]
    r = npr(x)[0].abs().mean(0).numpy()         # |residual| averaged over channels, in [0,2]
    axes[i,0].imshow(im); axes[i,0].set_ylabel(names[i], fontsize=9)
    axes[i,1].imshow(r, cmap="gray", vmin=0, vmax=0.4)
    axes[i,2].imshow(r[224:288, 224:288], cmap="gray", vmin=0, vmax=0.4, interpolation="nearest")
    print(p.name, "mean |npr|", r.mean())
for ax, t in zip(axes[0], ["Input (512$\\times$512)", "$|$NPR residual$|$ ($\\times$2.5)", "64$\\times$64 centre crop"]):
    ax.set_title(t, fontsize=8.5)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
print("saved", [p.name for p in picks])
