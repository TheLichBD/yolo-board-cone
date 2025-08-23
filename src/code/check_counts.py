from pathlib import Path
import collections
LBL=Path(r"/data/labels/all")
cnt=collections.Counter()
missing=0
for p in (Path(r"/data/images/all")).glob("*.*"):
    if p.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp"}: continue
    if not (LBL/f"{p.stem}.txt").exists(): missing+=1
for t in LBL.glob("*.txt"):
    for line in t.read_text(encoding="utf-8").splitlines():
        s=line.split()
        if s: cnt[s[0]]+=1
print("missing_labels:",missing)
print("counts_all:",dict(cnt)," (0=board, 1=cone)")
