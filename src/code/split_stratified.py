from pathlib import Path
import shutil,random
root=Path(r"/data")
ia,la=root/"images/all",root/"labels/all"
itr,ivr=root/"images/train",root/"images/val"
ltr,lvr=root/"labels/train",root/"labels/val"
for d in (itr,ivr,ltr,lvr):
    d.mkdir(parents=True,exist_ok=True)
    [f.unlink() for f in d.glob("*.*")]
exts={".jpg",".jpeg",".png",".bmp",".webp"}
pairs=[]
for im in ia.glob("*.*"):
    if im.suffix.lower() not in exts: continue
    lb=la/f"{im.stem}.txt"
    if lb.exists(): pairs.append((im,lb))
def classes(lb):
    s=set()
    for line in lb.read_text(encoding="utf-8").splitlines():
        z=line.split()
        if z: s.add(z[0])
    return s
b0,b1,b01=[],[],[]
for im,lb in pairs:
    cs=classes(lb)
    if cs=={"0"}: b0.append((im,lb))
    elif cs=={"1"}: b1.append((im,lb))
    else: b01.append((im,lb))
random.seed(42)
for b in (b0,b1,b01): random.shuffle(b)
def split(b,r=0.8):
    n=int(len(b)*r); return b[:n],b[n:]
t0,v0=split(b0); t1,v1=split(b1); t2,v2=split(b01)
train, val = t0+t1+t2, v0+v1+v2
random.shuffle(train); random.shuffle(val)
def cp(lst,split):
    for im,lb in lst:
        shutil.copy2(im,(itr if split=="train" else ivr)/im.name)
        shutil.copy2(lb,(ltr if split=="train" else lvr)/lb.name)
cp(train,"train"); cp(val,"val")
print("train:",len(train),"val:",len(val))
