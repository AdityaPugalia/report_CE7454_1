"""Grid-phase probe: does the detector depend on the parity of the pixel grid?"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/home/aditya/CE7454-assignment-1")
from model import build_model
from ablation import VARIANTS
from train import ImageDataset
dev = torch.device("cuda:0")
paths = sorted(Path("dev_phase/train").glob("*.jpg"))
labels = [ImageDataset.label_from_path(p) for p in paths]
_, val_paths = train_test_split(paths, test_size=0.2, stratify=labels, random_state=0)
y = np.array([ImageDataset.label_from_path(p) for p in val_paths])
def to_tensor(im):
    x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1)
    return x.float().div(255).sub(0.5).div(0.5)
class DS(Dataset):
    def __init__(self, tf): self.tf = tf
    def __len__(self): return len(val_paths)
    def __getitem__(self, i): return to_tensor(self.tf(Image.open(val_paths[i])))
@torch.no_grad()
def acc(model, tf):
    out = []
    for x in DataLoader(DS(tf), batch_size=50, num_workers=12):
        with torch.autocast("cuda"): out.append(model(x.to(dev)).float().argmax(1).cpu())
    return float((torch.cat(out).numpy() == y).mean())
def shift(dx, dy):  # crop so that the top-left corner moves by (dx, dy); size 508x508 for all
    return lambda im: im.crop((dx, dy, dx + 508, dy + 508))
def rot(k): return lambda im: im.rotate(90 * k, expand=True)
probes = {"shift(0,0)": shift(0,0), "shift(1,0)": shift(1,0), "shift(0,1)": shift(0,1), "shift(1,1)": shift(1,1),
          "shift(2,0)": shift(2,0), "shift(0,2)": shift(0,2), "shift(2,2)": shift(2,2), "shift(3,3)": shift(3,3),
          "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
          "rot90": rot(1), "rot180": rot(2), "rot270": rot(3),
          "crop_odd_offset_511": lambda im: im.crop((1,1,512,512)), "crop_even_offset_511": lambda im: im.crop((0,0,511,511))}
res = {}
for name in ["rgb_npr", "rgb_only", "npr_only"]:
    ck = "checkpoints/best.pt" if name == "rgb_npr" else f"checkpoints/{name}/best.pt"
    m = build_model(**VARIANTS[name]).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)["model"]); m.eval()
    res[name] = {}
    for k, tf in probes.items():
        res[name][k] = acc(m, tf); print(name, k, res[name][k], flush=True)
json.dump(res, open(Path(sys.argv[1]) / "probe.json", "w"), indent=2)
