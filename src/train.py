"""CV training (data split into 5 folds; we train 3 of them: folds 0,1,2). Usage: python train.py --model convnext_tiny --size 384 --epochs 6 --tag cnxt
Resumable: finished folds (models/<tag>_f<k>.pth) are skipped. Saves OOF probs to oof/<tag>.npy."""
import argparse, os, time, json
import cv2, numpy as np, pandas as pd, torch, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)
CACHE = r"E:\ADL\cache"
p = argparse.ArgumentParser()
p.add_argument("--model", default="convnext_tiny"); p.add_argument("--size", type=int, default=384)
p.add_argument("--epochs", type=int, default=6); p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--tag", default="cnxt")
p.add_argument("--folds", default="0,1,2"); p.add_argument("--workers", type=int, default=8)  # 3-fold; split is 5-way, pass 0,1,2,3,4 for all
a = p.parse_args()
os.makedirs(f"{CACHE}/models", exist_ok=True); os.makedirs(f"{CACHE}/oof", exist_ok=True)

df = pd.read_csv(f"{CACHE}/features.csv")
fp = f"{CACHE}/folds.csv"
if os.path.exists(fp):
    df["fold"] = pd.read_csv(fp).fold.values
else:  # group by hash so duplicates never straddle train/val
    df["fold"] = -1
    for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df.label, df.dhash)):
        df.loc[v, "fold"] = k
    df[["image_id", "label", "fold"]].to_csv(fp, index=False)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
S = a.size
train_tf = A.Compose([
    A.RandomResizedCrop((S, S), scale=(0.5, 1.0)), A.Transpose(p=.5), A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
    A.Affine(rotate=(-30, 30), p=.5), A.HueSaturationValue(10, 15, 10, p=.5),
    A.RandomBrightnessContrast(.15, .15, p=.5),
    A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(S // 16, S // 8), hole_width_range=(S // 16, S // 8), p=.3),
    A.Normalize(MEAN, STD), ToTensorV2()])
val_tf = A.Compose([A.Resize(S, S), A.Normalize(MEAN, STD), ToTensorV2()])


class DS(Dataset):
    def __init__(s, d, tf): s.ids, s.y, s.tf = d.image_id.values, d.label.values, tf
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        im = cv2.imread(f"{CACHE}/train_512/{s.ids[i]}")[:, :, ::-1]
        return s.tf(image=im)["image"], s.y[i]


def evaluate(m, dl):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl: out.append(m(x.cuda(non_blocking=True).to(memory_format=torch.channels_last)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    oof_path = f"{CACHE}/oof/{a.tag}.npy"
    oof = np.load(oof_path) if os.path.exists(oof_path) else np.zeros((len(df), 5), np.float32)
    for k in map(int, a.folds.split(",")):
        ck = f"{CACHE}/models/{a.tag}_f{k}.pth"
        if os.path.exists(ck): print("skip fold", k, flush=True); continue
        tr, va = df[df.fold != k], df[df.fold == k]
        tl = DataLoader(DS(tr, train_tf), a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
        vl = DataLoader(DS(va, val_tf), a.bs * 2, num_workers=a.workers, pin_memory=True)
        m = timm.create_model(a.model, pretrained=True, num_classes=5, drop_path_rate=0.1).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.05); sc = torch.amp.GradScaler()
        total = a.epochs * len(tl)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        best, hist = 0, []
        for ep in range(a.epochs):
            m.train(); t0 = time.time(); tot = 0
            for x, y in tl:
                x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last); y = y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16): loss = crit(m(x), y)
                opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); tot += loss.item()
            pr = evaluate(m, vl); acc = (pr.argmax(1) == va.label.values).mean()
            hist.append(dict(fold=k, epoch=ep, train_loss=tot / len(tl), val_acc=acc, secs=time.time() - t0))
            print(json.dumps(hist[-1]), flush=True)
            if acc > best: best = acc; torch.save(m.state_dict(), ck); oof[va.index.values] = pr
        pd.DataFrame(hist).to_csv(f"{CACHE}/models/{a.tag}_f{k}_hist.csv", index=False)
        np.save(oof_path, oof); print(f"fold {k} best {best:.4f}", flush=True)
        del m, opt, tl, vl; torch.cuda.empty_cache()
    msk = oof.sum(1) > 0
    print("OOF acc", (oof[msk].argmax(1) == df.label.values[msk]).mean(), "on", msk.sum(), flush=True)


if __name__ == "__main__":
    main()
