"""Extra analysis for the report: val confusion matrices, robustness, NPR stats."""
import io, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/home/aditya/CE7454-assignment-1")
from model import build_model, NPRResidual
from ablation import VARIANTS
from train import ImageDataset
torch.manual_seed(0)
dev = torch.device("cuda:0")
OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True, parents=True)

paths = sorted(Path("dev_phase/train").glob("*.jpg"))
labels = [ImageDataset.label_from_path(p) for p in paths]
_, val_paths = train_test_split(paths, test_size=0.2, stratify=labels, random_state=0)
val_labels = np.array([ImageDataset.label_from_path(p) for p in val_paths])
print("val", len(val_paths), np.bincount(val_labels))

def to_tensor(im): 
    x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
    return x.float().div(255).sub(0.5).div(0.5)

def load_variant(name):
    ck = "checkpoints/best.pt" if name=="rgb_npr" else f"checkpoints/{name}/best.pt"
    m = build_model(**VARIANTS[name]).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)["model"]); m.eval(); return m

from torch.utils.data import DataLoader, Dataset
class DS(Dataset):
    def __init__(self, tf): self.tf = tf
    def __len__(self): return len(val_paths)
    def __getitem__(self, i):
        im = Image.open(val_paths[i])
        if self.tf: im = self.tf(im)
        return to_tensor(im)

@torch.no_grad()
def predict(model, transform=None, bs=50):
    preds, probs = [], []
    dl = DataLoader(DS(transform), batch_size=bs, num_workers=16, shuffle=False)
    for x in dl:
        x = x.to(dev)
        with torch.autocast("cuda"):
            lg = model(x).float()
        preds.append(lg.argmax(1).cpu()); probs.append(lg.softmax(1).cpu())
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()

def jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f
def resize(s, back=True):
    def f(im):
        r = im.resize((s, s), Image.BILINEAR)
        return r.resize((512,512), Image.BILINEAR) if back else r
    return f
def center_crop(s):
    def f(im):
        o=(512-s)//2; return im.crop((o,o,o+s,o+s))
    return f
def blur(sig):
    from PIL import ImageFilter
    return lambda im: im.filter(ImageFilter.GaussianBlur(sig))
def noise(sd):
    def f(im):
        rng = np.random.default_rng(abs(hash(im.tobytes()[:4096])) % (2**32))
        a = np.array(im).astype(np.float32) + rng.normal(0, sd, (512,512,3))
        return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    return f
def hflip(im): return im.transpose(Image.FLIP_LEFT_RIGHT)

results = {}
for name in ["rgb_npr", "rgb_only", "npr_only"]:
    m = load_variant(name)
    p, pr = predict(m)
    cm = np.zeros((3,3), int)
    for t, q in zip(val_labels, p): cm[t, q] += 1
    conf = pr.max(1)
    results[name] = {"clean_acc": float((p==val_labels).mean()), "confusion": cm.tolist(),
                     "mean_conf": float(conf.mean()), "min_conf": float(conf.min()),
                     "frac_conf_below_0.99": float((conf<0.99).mean()),
                     "errors": [(val_paths[i].name, int(val_labels[i]), int(p[i])) for i in np.where(p!=val_labels)[0]]}
    print(name, results[name]["clean_acc"], cm.tolist(), "errors", results[name]["errors"])
    rob = {}
    for tname, tf in [("hflip", hflip), ("jpeg95", jpeg(95)), ("jpeg90", jpeg(90)), ("jpeg75", jpeg(75)), ("jpeg50", jpeg(50)),
                      ("resize256_back512", resize(256)), ("resize384_back512", resize(384)), ("resize256", resize(256, False)),
                      ("crop256", center_crop(256)), ("crop384", center_crop(384)),
                      ("blur0.5", blur(0.5)), ("blur1.0", blur(1.0)), ("noise2", noise(2)), ("noise5", noise(5))]:
        pp, _ = predict(m, tf)
        rob[tname] = float((pp==val_labels).mean()); print("  ", name, tname, rob[tname])
    results[name]["robustness"] = rob
    del m; torch.cuda.empty_cache()

# NPR residual statistics per class (energy + radial spectrum)
npr = NPRResidual()
stats = {c: [] for c in range(3)}; spec = {c: [] for c in range(3)}
for p, y in zip(val_paths[:900], val_labels[:900]):
    x = to_tensor(Image.open(p)).unsqueeze(0)
    r = npr(x)[0].mean(0)  # grayscale residual
    stats[y].append(float(r.abs().mean()))
    f = torch.fft.fftshift(torch.fft.fft2(x[0].mean(0))).abs().log1p()
    spec[y].append(f.numpy())
for c in range(3):
    print("class", c, "mean |NPR|", np.mean(stats[c]), "+-", np.std(stats[c]))
results["npr_abs_mean"] = {c: [float(np.mean(stats[c])), float(np.std(stats[c]))] for c in range(3)}
np.save(OUT/"mean_spectra.npy", np.stack([np.mean(spec[c],0) for c in range(3)]))
json.dump(results, open(OUT/"analysis.json","w"), indent=2)
print("saved")
