import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "results/"
p1 = json.load(open(S + "probe.json")); p2 = json.load(open(S + "probe2.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9})
COL = {"rgb_npr": "#2a78d6", "rgb_only": "#eb6834", "npr_only": "#1baf7a"}
LAB = {"rgb_npr": "RGB + NPR (ours)", "rgb_only": "RGB only", "npr_only": "NPR only"}
MK = {"rgb_npr": "o", "rgb_only": "s", "npr_only": "^"}
# diagonal shifts (d,d)
keys = [(0, "shift(0,0)", p1), (1, "shift(1,1)", p1), (2, "shift(2,2)", p1), (3, "shift(3,3)", p1), (4, "shift(4,4)", p2),
        (6, "shift(6,6)", p2), (8, "shift(8,8)", p2), (9, "shift(9,9)", p2), (16, "shift(16,16)", p2)]
fig, ax = plt.subplots(figsize=(3.3, 1.55))
for v in ["rgb_npr", "rgb_only", "npr_only"]:
    xs = [k[0] for k in keys]; ys = [100 * k[2][v][k[1]] for k in keys]
    ax.plot(xs, ys, color=COL[v], marker=MK[v], ms=3.5, lw=1.4, label=LAB[v])
ax.axhline(100/3, color="#52514e", lw=0.8, ls="--"); ax.text(16.3, 36, "chance", fontsize=7, ha="right", color="#52514e")
ax.set(xlabel="diagonal shift $d$ of the crop origin (pixels)", ylabel="val. accuracy (%)", xticks=[0,1,2,3,4,6,8,9,16], ylim=(25, 104))
ax.grid(alpha=0.3, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.42, 0.98))
plt.tight_layout(pad=0.3); plt.savefig(sys.argv[1], dpi=300, bbox_inches="tight"); print("saved")
