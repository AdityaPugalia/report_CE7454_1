import json, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8.5})
EP = re.compile(r"Epoch\s+\d+/\d+ \| train loss ([\d.]+) acc ([\d.]+) \| val loss ([\d.]+) acc ([\d.]+)")
cells = json.load(open("train.ipynb"))["cells"]
log = "".join("".join(o.get("text","")) for c in cells for o in c.get("outputs",[]))
rows = [tuple(map(float,m.groups())) for m in EP.finditer(log)]
tl, ta, vl, va = map(list, zip(*rows))
R = {"rgb_npr": {"train_losses": tl, "train_accs": ta, "val_losses": vl, "val_accs": va}}
for v in ["rgb_only", "npr_only"]: R[v] = json.load(open(f"results/{v}.json"))
COL = {"rgb_npr": "#2a78d6", "rgb_only": "#eb6834", "npr_only": "#1baf7a"}
LAB = {"rgb_npr": "RGB + NPR (ours)", "rgb_only": "RGB only", "npr_only": "NPR only"}
MK = {"rgb_npr": "o", "rgb_only": "s", "npr_only": "^"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 1.75))
for v, r in R.items():
    ep = range(1, len(r["val_losses"]) + 1)
    st = dict(color=COL[v], lw=1.6, marker=MK[v], ms=3.5, label=LAB[v])
    a1.plot(ep, r["val_losses"], **st)
    a2.plot(list(ep)[1:], [100*x for x in r["val_accs"][1:]], **st)
a1.set(yscale="log", xlabel="epoch", ylabel="val. cross-entropy", title="Validation loss")
a2.set(xlabel="epoch", ylabel="val. accuracy (%)", title="Validation accuracy (epochs 2+)", ylim=(98.8, 100.05))
for a in (a1, a2):
    a.grid(alpha=0.3, lw=0.5); a.spines[["top","right"]].set_visible(False)
    a.set_xticks(range(1, 14, 2))
a1.legend(frameon=False, fontsize=7)
plt.tight_layout(pad=0.4)
plt.savefig(sys.argv[1], dpi=300, bbox_inches="tight"); print("saved")
