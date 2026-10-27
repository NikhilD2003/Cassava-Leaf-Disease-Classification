"""Builds Cassava.ipynb from cell lists. Re-run after editing; then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""# Cassava Leaf Disease Classification
Local end-to-end notebook: **EDA -> cross-validated training -> model analysis -> TTA ensemble inference**.
Runs on E:\\ADL (data in `data/`, resized cache + stats from `src/prep.py`, training in `src/train.py`).""")

md("## 1. Setup")
code("""
import sys, torch, torchvision
print(sys.executable, "| torch", torch.__version__, "| torchvision", torchvision.__version__)
assert "Python310" in sys.executable, "Wrong kernel: pick 'adl310' (Python 3.10) - other kernels have mismatched torch/torchvision"
import torchvision.ops  # fails with 'torchvision::nms does not exist' if torch/torchvision versions mismatch
""")
code("""
import os, json, warnings, numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
ROOT, CACHE = r"E:\\ADL\\data", r"E:\\ADL\\cache"
LABELS = json.load(open(f"{ROOT}/label_num_to_disease_map.json"))
LABELS = {int(k): v for k, v in LABELS.items()}
SHORT = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}
df = pd.read_csv(f"{CACHE}/features.csv")
df["cls"] = df.label.map(SHORT)
print(df.shape); df.head()
""")

md("## 2. EDA")
md("### 2.1 Class balance")
code("""
vc = df.cls.value_counts().reindex(SHORT.values())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=vc.index, y=vc.values, ax=ax[0], palette="viridis")
for i, v in enumerate(vc.values): ax[0].text(i, v, f"{v}\\n({v/len(df):.1%})", ha="center", va="bottom")
ax[0].set_title("Images per class"); ax[0].set_ylim(0, vc.max()*1.18)
ax[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=sns.color_palette("viridis", 5)); ax[1].set_title("Share")
plt.tight_layout(); plt.show()
print(f"Imbalance ratio majority/minority: {vc.max()/vc.min():.1f}x | majority-class baseline accuracy: {vc.max()/len(df):.3f}")
""")
md("""**Takeaway:** CMD is ~61% of the data, so always predicting CMD already scores ~0.61. Overall accuracy hides weak minority classes - we track per-class recall/F1 later.""")

md("### 2.2 Image geometry")
code("""
df["wh"] = df.w.astype(str) + "x" + df.h.astype(str)
print(df.wh.value_counts().head(), "\\n")
print("aspect ratio unique:", df.eval("w/h").round(3).unique())
""")

md("### 2.3 Sample images per class")
code("""
fig, axes = plt.subplots(5, 6, figsize=(18, 13))
for r, k in enumerate(SHORT):
    s = df[df.label == k].sample(6, random_state=1)
    for c, (_, row) in enumerate(s.iterrows()):
        axes[r, c].imshow(cv2.imread(f"{CACHE}/train_512/{row.image_id}")[:, :, ::-1]); axes[r, c].axis("off")
        if c == 0: axes[r, c].set_title(LABELS[k], loc="left", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
""")

md("### 2.4 Colour, brightness, contrast, sharpness by class")
code("""
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for a, col in zip(ax.ravel(), ["r", "g", "b", "bright", "contrast", "sharp"]):
    d = df[df[col] < df[col].quantile(.995)]
    sns.violinplot(data=d, x="cls", y=col, order=list(SHORT.values()), ax=a, inner="quartile", palette="viridis", cut=0)
    a.set_title(col); a.set_xlabel("")
plt.tight_layout(); plt.show()
print(df.groupby("cls")[["r", "g", "b", "bright", "contrast", "sharp"]].mean().round(1))
""")
md("""Mean-colour and sharpness distributions overlap heavily between classes: the model must learn **local lesion texture/pattern**, not global colour. This justifies high input resolution and texture-preserving augmentation (no heavy blur).""")

md("### 2.5 Colour-channel correlation / feature separability")
code("""
from sklearn.decomposition import PCA
f = ["r", "g", "b", "bright", "contrast", "sharp"]
z = (df[f] - df[f].mean()) / df[f].std()
p = PCA(2).fit_transform(z)
plt.figure(figsize=(7, 5)); sns.scatterplot(x=p[:, 0], y=p[:, 1], hue=df.cls, hue_order=list(SHORT.values()), s=6, alpha=.5)
plt.title("PCA of handcrafted image stats (classes are NOT separable this way)"); plt.show()
sns.heatmap(df[f].corr(), annot=True, fmt=".2f", cmap="vlag", center=0); plt.show()
""")

md("### 2.6 Duplicates and label noise (perceptual hash)")
code("""
g = df.groupby("dhash")
dup = g.filter(lambda x: len(x) > 1)
print(f"{df.dhash.duplicated(keep=False).sum()} images share a hash with another image "
      f"({dup.dhash.nunique()} groups); {df.dhash.duplicated().sum()} are redundant copies")
conf = g.filter(lambda x: len(x) > 1 and x.label.nunique() > 1)
print(f"{conf.dhash.nunique()} duplicate groups carry CONFLICTING labels ({len(conf)} images) -> label noise")
sizes = dup.groupby("dhash").size()
print("group size distribution:", sizes.value_counts().sort_index().to_dict())
""")
code("""
cg = conf.groupby("dhash")
show = list(cg.groups.keys())[:6]
if not show: print("no exact-hash duplicates with conflicting labels in this data")
else:
    fig, axes = plt.subplots(len(show), 3, figsize=(10, 3.2*len(show)), squeeze=False)
    for r, h in enumerate(show):
        for c, (_, row) in enumerate(cg.get_group(h).head(3).iterrows()):
            axes[r, c].imshow(cv2.imread(f"{CACHE}/train_512/{row.image_id}")[:, :, ::-1]); axes[r, c].axis("off")
            axes[r, c].set_title(f"label={SHORT[row.label]}")
    plt.suptitle("Near-identical images with different labels", y=1.0); plt.tight_layout(); plt.show()
""")
md("""**Why this matters:** (1) duplicates split across train/val **leak** and inflate CV scores, so folds are built with `StratifiedGroupKFold` grouped by hash; (2) conflicting labels set a ceiling on achievable accuracy and motivate label smoothing.""")

md("### 2.7 Test set")
code("""
t = os.listdir(f"{ROOT}/test_images"); print("local test images:", t, "- the real test set is hidden on Kaggle; we score via cross-validation.")
""")

md("## 3. Training setup")
md("""Training is done by `train.py` (kept out of the notebook so a kernel crash cannot lose hours of GPU time; it is resumable per fold).

| Item | Choice | Why |
|---|---|---|
| Backbone | `convnext_tiny` (ImageNet-pretrained, timm) | best accuracy/throughput on the 8 GB RTX 4060 in our benchmark (~74 img/s vs 16-21 for EfficientNet-B3/B4) |
| Input | 384x384 from a 512x384 cache | lesions are small; resolution matters |
| Folds | data split into 5 `StratifiedGroupKFold` folds grouped by perceptual hash; **3 folds (0-2) trained** (time budget), giving a 3-model ensemble | no duplicate leakage between train/val |
| Loss | cross-entropy, label smoothing 0.1 | label noise found in 2.6 |
| Optimiser | AdamW, lr 1e-4, wd 0.05, one-cycle cosine, 6 epochs | |
| Precision | fp16 autocast + channels_last | speed / memory |
| Augmentation | RandomResizedCrop, transpose/flips, affine rotate, HSV, brightness/contrast, CoarseDropout | leaves are orientation-free; occlusion robustness |""")
code("""
print(open(r"E:\\ADL\\src\\train.py", encoding="utf8").read())
""")

md("## 4. Cross-validated results")
code("""
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
TAG = "cnxt"
hist = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{CACHE}/models/{TAG}_f*_hist.csv"))])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k, h in hist.groupby("fold"):
    ax[0].plot(h.epoch, h.train_loss, label=f"fold {k}"); ax[1].plot(h.epoch, h.val_acc, label=f"fold {k}")
ax[0].set_title("Train loss (label-smoothed CE)"); ax[1].set_title("Validation accuracy")
for a in ax: a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.show()
print(hist.groupby("fold").val_acc.max().round(4).to_dict(), "| mean best:", round(hist.groupby("fold").val_acc.max().mean(), 4))
print(f"seconds/epoch (mean): {hist.secs.mean():.0f}")
""")
code("""
oof = np.load(f"{CACHE}/oof/{TAG}.npy"); keep = oof.sum(1) > 0  # folds 3-4 untrained -> zero rows
oof = oof[keep]; d = df[keep].reset_index(drop=True); y = d.label.values; pred = oof.argmax(1)
print(f"OOF accuracy: {accuracy_score(y, pred):.4f} on {keep.sum()} images (majority baseline {d.label.value_counts(normalize=True).max():.3f})\\n")
print(classification_report(y, pred, target_names=list(SHORT.values()), digits=3))
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SHORT.values(), yticklabels=SHORT.values(), ax=ax[0]); ax[0].set_title("Confusion matrix (counts)")
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", xticklabels=SHORT.values(), yticklabels=SHORT.values(), ax=ax[1]); ax[1].set_title("Row-normalised (recall)")
for a in ax: a.set_xlabel("predicted"); a.set_ylabel("true")
plt.tight_layout(); plt.show()
""")

md("### 4.1 Error analysis")
code("""
conf = oof.max(1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(conf[pred == y], bins=30, alpha=.7, label="correct"); ax[0].hist(conf[pred != y], bins=30, alpha=.7, label="wrong")
ax[0].set_title("Prediction confidence"); ax[0].legend()
ok = pd.Series(pred == y).groupby(pd.cut(conf, [0, .5, .6, .7, .8, .9, 1.0])).mean()
ok.plot.bar(ax=ax[1], rot=0); ax[1].set_title("Accuracy by confidence bin (calibration)"); plt.tight_layout(); plt.show()

wrong = np.where(pred != y)[0]; wrong = wrong[np.argsort(-conf[wrong])][:12]
fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
for a, i in zip(axes.ravel(), wrong):
    a.imshow(cv2.imread(f"{CACHE}/train_512/{d.image_id[i]}")[:, :, ::-1]); a.axis("off")
    a.set_title(f"true {SHORT[y[i]]} / pred {SHORT[pred[i]]}\\nconf {conf[i]:.2f}", fontsize=9)
plt.suptitle("Most confident mistakes - often mislabelled or ambiguous images", y=1.0); plt.tight_layout(); plt.show()
dupe = d.dhash.duplicated(keep=False).values
print(f"accuracy on images with a duplicate: {accuracy_score(y[dupe], pred[dupe]):.4f} | without: {accuracy_score(y[~dupe], pred[~dupe]):.4f}")
""")

md("## 5. Final inference: 3-fold ensemble + test-time augmentation")
md("""Each of the 3 fold models predicts 4 views of every image (original, horizontal flip, vertical flip, transpose); the 12 softmax outputs are averaged. Fold models see different training data, so averaging them reduces variance beyond what a single model can offer.""")
code("""
import torch, timm, albumentations as A
from albumentations.pytorch import ToTensorV2
DEV, SIZE, ARCH = "cuda" if torch.cuda.is_available() else "cpu", 384, "convnext_tiny"
tf = A.Compose([A.Resize(SIZE, SIZE), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ToTensorV2()])
models = []
for f in sorted(glob.glob(f"{CACHE}/models/{TAG}_f?.pth")):
    m = timm.create_model(ARCH, pretrained=False, num_classes=5)
    m.load_state_dict(torch.load(f, map_location="cpu")); models.append(m.to(DEV).eval().to(memory_format=torch.channels_last))
print(len(models), "fold models loaded on", DEV)

@torch.no_grad()
def predict(x):  # x: (B,3,H,W) -> mean softmax over models x 4 TTA views
    views = [x, x.flip(3), x.flip(2), x.transpose(2, 3)]
    out = 0
    for m in models:
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            for v in views: out = out + m(v.contiguous(memory_format=torch.channels_last)).float().softmax(1)
    return (out / (len(models) * len(views))).cpu().numpy()
""")
code("""
TEST = f"{ROOT}/test_images"
ids = sorted(os.listdir(TEST)); probs = []
for i in range(0, len(ids), 16):
    b = [tf(image=cv2.imread(f"{TEST}/{n}")[:, :, ::-1])["image"] for n in ids[i:i+16]]
    probs.append(predict(torch.stack(b).to(DEV)))
probs = np.concatenate(probs)
sub = pd.DataFrame({"image_id": ids, "label": probs.argmax(1)})
sub.to_csv(r"E:\\ADL\\submission.csv", index=False)
print(sub.merge(pd.DataFrame(probs, columns=list(SHORT.values())).assign(image_id=ids)).round(3).to_string(index=False))
print("saved E:\\\\ADL\\\\submission.csv (same format as sample_submission.csv)")
""")
md("""## 6. Summary
* Data are imbalanced (CMD ~61%) and contain duplicates with conflicting labels, so folds are grouped by hash and loss uses label smoothing.
* A 3-fold ConvNeXt-Tiny (folds 0-2 of a 5-way split) at 384px gives the OOF accuracy reported in section 4; remaining errors concentrate in visually similar classes (see confusion matrix) and in near-duplicate label noise.
* The submitted prediction is a 3-model x 4-view average, written to `submission.csv`.""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "adl310", "display_name": "adl310", "language": "python"}
nbf.write(nb, r"E:\ADL\notebooks\Cassava.ipynb")
print("cells:", len(C))
