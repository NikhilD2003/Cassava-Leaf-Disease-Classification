"""One pass over train_images: resize to 512x384 cache + per-image stats + dhash (for EDA/dedup)."""
import os, sys
import cv2, numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = r"E:\ADL\data"
CACHE = r"E:\ADL\cache\train_512"
os.makedirs(CACHE, exist_ok=True)


def work(img_id):
    im = cv2.imread(os.path.join(ROOT, "train_images", img_id))
    h, w = im.shape[:2]
    rgb = im[:, :, ::-1]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).ravel()), 2)
    r, gg, b = (rgb[:, :, i].mean() for i in range(3))
    s = cv2.resize(im, (512, 384), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(CACHE, img_id), s, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dict(image_id=img_id, h=h, w=w, r=r, g=gg, b=b, bright=g.mean(),
                contrast=g.std(), sharp=cv2.Laplacian(g, cv2.CV_64F).var(), dhash=str(dh))


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    with Pool(12) as p:
        rows = p.map(work, df.image_id.tolist(), chunksize=64)
    out = df.merge(pd.DataFrame(rows), on="image_id")
    out.to_csv(r"E:\ADL\cache\features.csv", index=False)
    print(out.shape)
