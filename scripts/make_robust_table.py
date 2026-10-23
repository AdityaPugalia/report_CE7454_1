import re, sys
log = open("results/robustness.log").read()
R = {}
for m in re.finditer(r"^\s+(\w+) (\S+) ([\d.]+)$", log, re.M):
    R.setdefault(m.group(1), {})[m.group(2)] = 100 * float(m.group(3))
rows = [("None (clean)", None), ("JPEG q=95", "jpeg95"), ("JPEG q=90", "jpeg90"), ("JPEG q=75", "jpeg75"), ("JPEG q=50", "jpeg50"),
        ("Resize $512{\\to}256{\\to}512$", "resize256_back512"),
        ("Resize $512{\\to}256$", "resize256"), ("Centre crop $384^2$", "crop384"), ("Centre crop $256^2$", "crop256"),
        ("Gaussian blur $\\sigma{=}1$", "blur1.0"),
        ("Gaussian noise $\\sigma{=}2$", "noise2"), ("Gaussian noise $\\sigma{=}5$", "noise5")]
clean = {"rgb_npr": 100.0, "rgb_only": 99.97, "npr_only": 99.97}
out = ["\\begin{table}[t]", "  \\centering\\small", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{@{}lccc@{}}", "    \\toprule",
       "    Perturbation & RGB\\,+\\,NPR & RGB only & NPR only \\\\", "    \\midrule"]
def f(v): return "--" if v is None else (f"{v:.1f}" if v < 99.95 else "100")
for name, key in rows:
    vals = [clean[v] if key is None else R.get(v, {}).get(key) for v in ["rgb_npr", "rgb_only", "npr_only"]]
    out.append(f"    {name} & " + " & ".join(f(v) for v in vals) + " \\\\")
out += ["    \\bottomrule", "  \\end{tabular}",
        "  \\caption{Post-hoc robustness: validation accuracy (\\%) of the three trained models when the validation images are perturbed before inference. No model is retrained. Resizes use bilinear filtering; noise is i.i.d.\\ per pixel in 8-bit units.}",
        "  \\label{tab:robust}", "\\end{table}"]
open("fig/robust_table.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out[7:22]))
