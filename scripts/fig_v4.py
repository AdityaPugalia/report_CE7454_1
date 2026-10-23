"""Figures for the v4 report. Run from report/:  python scripts/fig_v4.py
Reads results/v4_analysis.json, results/spectrum_zoom.npz, ../results/v4_aug_scratch.json; writes fig/*.pdf, fig/failures.png."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
BASE, V4, INK, MUTED = "#eb6834", "#2a78d6", "#2b2b2a", "#8a8984"
A = json.load(open("results/v4_analysis.json"))


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.5)


# ---------------------------------------------------------------- Fig. 2: diagnosis (phase drift + spectral peaks)
Z = np.load("results/spectrum_zoom.npz")
fig = plt.figure(figsize=(3.35, 1.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.1)
ax = fig.add_subplot(gs[0])
x = np.arange(0, 513)
ax.plot(x, np.zeros_like(x), color=BASE, lw=1.4)
ax.plot(x, (x - x * 508 / 512), color=V4, lw=1.4)
ax.text(10, 0.35, "train / dev", fontsize=6.3, color=BASE)
ax.text(20, 3.2, "test\n" + r"($\times$512/508)", fontsize=6.3, color=V4, va="bottom")
ax.set(xlabel="column $x$ (px)", ylabel="lattice offset (px)", xlim=(0, 512), ylim=(-0.3, 4.6), xticks=[0, 256, 512], yticks=[0, 2, 4])
tidy(ax); ax.set_title("(a)", loc="left", fontsize=7.5)
W = 7  # half window (bins)
for i, (k, t) in enumerate([("train_B", "(b) train B"), ("sim_B", "(c) simulated"), ("test", "(d) test")]):
    ax = fig.add_subplot(gs[i + 1]); im = np.log(Z[k + "_p4"])[14 - W:14 + W + 1, 14 - W:14 + W + 1]
    ax.imshow(im, cmap="magma", vmin=im.min(), vmax=im.max(), interpolation="nearest", extent=(128 - W - .5, 128 + W + .5, 128 + W + .5, 128 - W - .5))
    ax.axhline(128, color="w", lw=0.4, ls=":"); ax.axvline(128, color="w", lw=0.4, ls=":")
    ax.set_xticks([122, 128, 134]); ax.tick_params(labelsize=5.5, length=2)
    ax.set_yticks([]); ax.set_title(t, fontsize=6.5, loc="left", pad=2)
    if i == 1: ax.set_xlabel("spectrum bin", fontsize=6.5, labelpad=1)
plt.savefig("fig/diagnosis.pdf", bbox_inches="tight", pad_inches=0.01); plt.close()

# ---------------------------------------------------------------- Fig. 3: probes baseline vs v4
P = A["probes"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.35, 1.75), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.85})
d = list(range(9))
for name, col, lab in (("baseline", BASE, "baseline"), ("v4", V4, "ours")):
    a1.plot(d, [100 * P[name][f"shift {k}"] for k in d], color=col, marker="o", ms=2.8, lw=1.3, label=lab)
a1.axhline(100 / 3, color=MUTED, lw=0.7, ls="--"); a1.text(4, 36.5, "chance", fontsize=6, ha="center", color=MUTED)
a1.set(xlabel="diagonal crop shift $d$ (px)", ylabel="val. accuracy (%)", xticks=d, ylim=(25, 104))
a1.legend(frameon=False, fontsize=6.3, loc="center", bbox_to_anchor=(0.5, 0.6)); tidy(a1)
a1.set_title("(a) grid phase", loc="left", fontsize=7.5)
keys = [("simulated test", "simulated test"), ("hflip", "h-flip"), ("rot90", "rot 90°"), ("rescale 0.9", "rescale 0.9"),
        ("rescale 1.1", "rescale 1.1"), ("jpeg q75", "JPEG q75"), ("jpeg q50", "JPEG q50"), ("resize 256", "resize 256"),
        ("noise 5", r"noise $\sigma$=5"), ("blur 1", r"blur $\sigma$=1")]
yy = np.arange(len(keys))[::-1]
for name, col, off in (("baseline", BASE, 0.18), ("v4", V4, -0.18)):
    a2.barh(yy + off, [100 * P[name][k] for k, _ in keys], height=0.34, color=col, edgecolor="white", linewidth=0.4)
a2.axvline(100 / 3, color=MUTED, lw=0.7, ls="--")
a2.set_yticks(yy); a2.set_yticklabels([l for _, l in keys], fontsize=6.3); a2.set_xlim(0, 100); a2.set_xticks([0, 50, 100])
a2.set_xlabel("val. accuracy (%)"); a2.spines[["top", "right"]].set_visible(False); a2.grid(axis="x", alpha=0.25, lw=0.5)
a2.set_title("(b) perturbations", loc="left", fontsize=7.5)
plt.savefig("fig/probes.pdf", bbox_inches="tight", pad_inches=0.01); plt.close()

# ---------------------------------------------------------------- Fig. 4: v4 training curves
H = json.load(open("../results/v4_aug_scratch.json"))["history"]
e = [h["epoch"] for h in H]
fig, ax = plt.subplots(figsize=(3.35, 1.35))
for k, col, lab, ls in (("train_acc", MUTED, "train (augmented crops)", "-"), ("harsh_acc", V4, "harsh val (selection)", "-"),
                        ("sim_acc", "#1baf7a", "simulated-test val", "-")):
    ax.plot(e, [100 * h[k] for h in H], color=col, lw=1.3, ls=ls, label=lab)
best = max(H, key=lambda h: h["harsh_acc"]); ax.plot(best["epoch"], 100 * best["harsh_acc"], "o", ms=4, mfc="none", mec=INK, mew=0.8)
ax.annotate("saved (ep. 26)", (best["epoch"], 100 * best["harsh_acc"]), (19.5, 80), fontsize=6.3, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
ax.set(xlabel="epoch (152k images each)", ylabel="accuracy (%)", ylim=(50, 101), xlim=(1, 30))
ax.legend(frameon=False, fontsize=6.3, loc="lower right", bbox_to_anchor=(1.0, 0.02)); tidy(ax)
plt.savefig("fig/curves_v4.pdf", bbox_inches="tight", pad_inches=0.01); plt.close()

# ---------------------------------------------------------------- Fig. 5: most confident dev errors
errs = A["v4_errors"][:4]
names = {0: "real", 1: "A", 2: "B"}
fig, axs = plt.subplots(1, 4, figsize=(3.35, 1.05))
for ax, r in zip(axs, errs):
    im = Image.open(f"../dev_phase/dev/{r['name']}").convert("RGB")
    ax.imshow(im.resize((192, 192), Image.LANCZOS)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
    ax.set_title(f"{names[r['true']]} $\\to$ {names[r['pred']]}", fontsize=6.5, pad=2)
plt.subplots_adjust(wspace=0.12)
plt.savefig("fig/failures.pdf", bbox_inches="tight", pad_inches=0.01, dpi=200); plt.close()
print("figures written")
