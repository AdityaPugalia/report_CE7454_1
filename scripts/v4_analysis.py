"""Analysis for the v4 report: baseline (best.pt) vs v4 and two v4 ablations, all single pass.

Run from the repository root:  python report/scripts/v4_analysis.py report/results
Writes <out>/v4_analysis.json (probes, dev confusion, texture analysis, failure list) and <out>/spectrum_zoom.npz.
"""
import io, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts_v3"))
from model import build_model  # noqa: E402
from train import ImageDataset  # noqa: E402
from make_aug_dataset import texture  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
paths = sorted((ROOT / "dev_phase/train").glob("*.jpg"))
labels = [ImageDataset.label_from_path(p) for p in paths]
_, val_paths = train_test_split(paths, test_size=0.2, stratify=labels, random_state=0)
y_val = np.array([ImageDataset.label_from_path(p) for p in val_paths])
dev_paths = sorted((ROOT / "dev_phase/dev").glob("*.jpg"), key=lambda p: int(p.stem))
gold = json.load(open(ROOT / "results/dev_submission/predictions.json"))  # scored 100% on CodaBench
y_dev = np.array([gold[p.name] for p in dev_paths])

MODELS = {  # name: (checkpoint, antialias)
    "baseline": ("checkpoints/best.pt", False),
    "v4": ("checkpoints/v4_aug_scratch/best.pt", True),
    "v3b2_matched": ("checkpoints/v3b2_matched_crop256_cont/best.pt", True),
    "v5_crop128": ("checkpoints/v5_crop128_bal/last.pt", True),
}


def to_tensor(im):
    return torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)


def jpeg(q, ss=0):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q, subsampling=ss); return Image.open(b).convert("RGB")
    return f


def simulate(im):  # organisers' processing, reproduced on held-out images
    return jpeg(90, 2)(im.crop((2, 2, 510, 510)).resize((512, 512), Image.BICUBIC))


def shift(d):
    return lambda im: im.crop((d, d, d + 504, d + 504))


def rescale(s):
    return lambda im: im.resize((round(512 * s),) * 2, Image.BICUBIC)


def noise(sd):
    def f(im):
        rng = np.random.default_rng(0)
        return Image.fromarray(np.clip(np.array(im).astype(np.float32) + rng.normal(0, sd, (512, 512, 3)), 0, 255).astype(np.uint8))
    return f


PROBES = {"clean": None, "simulated test": simulate,
          **{f"shift {d}": shift(d) for d in range(9)},
          "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), "rot90": lambda im: im.rotate(90),
          "jpeg q90 4:2:0": jpeg(90, 2), "jpeg q75": jpeg(75), "jpeg q50": jpeg(50),
          "rescale 0.9": rescale(0.9), "rescale 1.1": rescale(1.1), "resize 256": rescale(0.5),
          "blur 1": lambda im: im.filter(ImageFilter.GaussianBlur(1)), "noise 5": noise(5)}


class DS(Dataset):
    def __init__(self, ps, tf): self.ps, self.tf = ps, tf
    def __len__(self): return len(self.ps)
    def __getitem__(self, i):
        im = Image.open(self.ps[i]).convert("RGB")
        return to_tensor(self.tf(im) if self.tf else im)


@torch.no_grad()
def probs(model, ps, tf=None, bs=16):
    out = []
    for x in DataLoader(DS(ps, tf), batch_size=bs, num_workers=16):
        with torch.autocast("cuda"):
            out.append(model(x.to(dev)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


res = {"probes": {}, "dev": {}}
for name, (ck, aa) in MODELS.items():
    m = build_model(num_classes=3, antialias=aa, use_spectrum=False).to(dev)
    m.load_state_dict(torch.load(ROOT / ck, map_location=dev)["model"]); m.eval()
    res["probes"][name] = {}
    probe_set = PROBES if name in ("baseline", "v4") else {k: PROBES[k] for k in ("clean", "simulated test")}
    for k, tf in probe_set.items():
        acc = float((probs(m, val_paths, tf).argmax(1) == y_val).mean())
        res["probes"][name][k] = acc; print(name, k, round(acc, 4), flush=True)
    P = probs(m, dev_paths); pred = P.argmax(1)
    cm = np.zeros((3, 3), int)
    for a, b in zip(y_dev, pred): cm[a, b] += 1
    res["dev"][name] = {"acc": float((pred == y_dev).mean()), "confusion": cm.tolist()}
    print(name, "dev", res["dev"][name], flush=True)
    if name == "v4":
        tex = np.array([texture(p) for p in dev_paths])
        q = np.quantile(tex[y_dev == 0], [0.25, 0.5, 0.75])
        bins = np.digitize(tex, q)
        res["v4_real_recall_by_texture_quartile"] = [float((pred[(y_dev == 0) & (bins == b)] == 0).mean()) for b in range(4)]
        res["v4_fake_recall_by_texture_quartile"] = [float((pred[(y_dev > 0) & (bins == b)] == y_dev[(y_dev > 0) & (bins == b)]).mean()) for b in range(4)]
        res["texture_quartiles_real"] = q.tolist()
        err = np.where(pred != y_dev)[0]
        res["v4_errors"] = sorted([{"name": dev_paths[i].name, "true": int(y_dev[i]), "pred": int(pred[i]),
                                     "conf": float(P[i].max()), "texture": float(tex[i])} for i in err], key=lambda r: -r["conf"])
    del m; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "v4_analysis.json", "w"), indent=2)

# spectrum zoom: mean |FFT| of the luma high-pass around the period-8 / period-4 diagonal sites
def spec(p, tf=None):
    im = Image.open(p).convert("RGB"); im = tf(im) if tf else im
    yl = np.asarray(im.convert("L")).astype(np.float32)
    hp = yl - (np.roll(yl, 1, 0) + np.roll(yl, -1, 0) + np.roll(yl, 1, 1) + np.roll(yl, -1, 1)) / 4
    return np.abs(np.fft.fftshift(np.fft.fft2(hp)))
fakes = [p for p, l in zip(val_paths, y_val) if l == 2][:400]
test = sorted((ROOT / "test_images").glob("*.jpg"))[:1500]
S = {"train_B": np.mean([spec(p) for p in fakes], 0), "sim_B": np.mean([spec(p, simulate) for p in fakes], 0),
     "test": np.mean([spec(p) for p in test], 0)}
np.savez_compressed(OUT / "spectrum_zoom.npz", **{k: v[256 + 50:256 + 79, 256 + 50:256 + 79] for k, v in S.items()},
                    **{k + "_p4": v[256 + 114:256 + 143, 256 + 114:256 + 143] for k, v in S.items()})
print("done")
