import base64, csv, json, random, shutil, threading, time
from pathlib import Path
import cv2, numpy as np, yaml
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

W = Path(__file__).parent / "workspace"
IMG, LBL, DS, WTS, RUNS = (W / d for d in ("images", "labels", "dataset", "weights", "runs"))
for d in (IMG, LBL, DS, WTS, RUNS): d.mkdir(parents=True, exist_ok=True)
FRONT = Path(__file__).parent.parent / "frontend"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["Empty", "Occupied", "Uncertain"]
BGR = {0: (94, 197, 34), 1: (68, 68, 239), 2: (8, 179, 234)}
app = FastAPI(title="SmartPark")

def safe(n): return Path(n).name

# ---------- Module 1: annotation ----------
@app.post("/api/import")
async def import_images(files: list[UploadFile] = File(...)):
    added = skipped = 0
    for f in files:
        name = safe(f.filename)
        if Path(name).suffix.lower() not in EXT: continue
        if (IMG / name).exists(): skipped += 1; continue  # never overwrite
        (IMG / name).write_bytes(await f.read()); added += 1
    return {"added": added, "skipped": skipped}

def lbl_path(n): return LBL / (Path(n).stem + ".txt")
def read_boxes(n):
    p, out = lbl_path(n), []
    if p.exists():
        for ln in p.read_text().splitlines():
            s = ln.split()
            if len(s) == 5: out.append(dict(zip(("cls", "x", "y", "w", "h"), [int(s[0])] + [float(v) for v in s[1:]])))
    return out

@app.get("/api/images")
def images(filter: str = "all"):
    res = [{"name": p.name, "labeled": len(read_boxes(p.name)) > 0}
           for p in sorted(IMG.iterdir()) if p.suffix.lower() in EXT]
    if filter == "labeled": res = [r for r in res if r["labeled"]]
    if filter == "unlabeled": res = [r for r in res if not r["labeled"]]
    return res

@app.get("/api/image/{name}")
def image(name: str): return FileResponse(IMG / safe(name))

@app.get("/api/labels/{name}")
def get_labels(name: str): return read_boxes(safe(name))

@app.put("/api/labels/{name}")
def put_labels(name: str, boxes: list[dict]):  # auto-save as normalized YOLO txt
    lbl_path(safe(name)).write_text("".join(
        f"{b['cls']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}\n" for b in boxes))
    return {"ok": True}

def build_dataset(split=0.8):
    shutil.rmtree(DS, ignore_errors=True)
    names = [p.name for p in IMG.iterdir() if p.suffix.lower() in EXT and read_boxes(p.name)]
    if len(names) < 2: raise HTTPException(400, "Label at least 2 images before training.")
    random.Random(42).shuffle(names)
    k = max(1, int(len(names) * split)) if len(names) > 1 else 1
    k = min(k, len(names) - 1)
    for part, subset in (("train", names[:k]), ("val", names[k:])):
        (DS / "images" / part).mkdir(parents=True); (DS / "labels" / part).mkdir(parents=True)
        for n in subset:
            shutil.copy(IMG / n, DS / "images" / part / n)
            shutil.copy(lbl_path(n), DS / "labels" / part / (Path(n).stem + ".txt"))
    cfg = {"path": str(DS.resolve()), "train": "images/train", "val": "images/val",
           "names": {i: n for i, n in enumerate(NAMES)}}
    (DS / "data.yaml").write_text(yaml.safe_dump(cfg))
    return DS / "data.yaml"

# ---------- Weights library ----------
@app.get("/api/weights")
def weights(): return sorted(p.name for p in WTS.glob("*.pt"))

@app.post("/api/weights")
async def upload_weights(file: UploadFile = File(...)):
    (WTS / safe(file.filename)).write_bytes(await file.read()); return {"ok": True}

# ---------- Module 2: training ----------
ST = {"state": "idle", "epoch": 0, "epochs": 0, "log": [], "run": None, "error": None}
TRAINER = {"t": None}

def log(m): ST["log"].append(f"[{time.strftime('%H:%M:%S')}] {m}"); ST["log"] = ST["log"][-500:]

def worker(p):
    try:
        import torch
        from ultralytics import YOLO
        data = build_dataset(); log(f"Dataset ready: {data}")
        base = p["base"]; model = YOLO(str(WTS / base) if (WTS / base).exists() else base)
        dev = 0 if torch.cuda.is_available() else "cpu"; log(f"Device: {'CUDA GPU' if dev == 0 else 'CPU'}")
        def on_start(t): TRAINER["t"] = t
        def on_epoch(t):
            ST["epoch"] = t.epoch + 1
            log(f"Epoch {t.epoch + 1}/{t.epochs} done | " + ", ".join(f"{k}={float(v):.3f}" for k, v in t.metrics.items()
                                                                    if "mAP50(B)" in k or "precision" in k or "recall" in k))
            if ST["state"] == "cancelling": t.stop = True
        model.add_callback("on_pretrain_routine_start", on_start)
        model.add_callback("on_train_epoch_end", on_epoch)
        keys = ["epochs", "batch", "imgsz", "lr0", "optimizer", "hsv_h", "hsv_s", "hsv_v", "degrees", "shear",
                "perspective", "fliplr", "flipud", "mosaic"]
        args = {k: p[k] for k in keys if k in p}
        args.update(data=str(data), project=str(RUNS.resolve()), name=p["name"], exist_ok=True, device=dev,
                    plots=True, verbose=False)
        if p.get("tune"):
            log("Auto-tuning hyperparameters (this is slow)...")
            model.tune(data=str(data), epochs=min(args["epochs"], 10), iterations=10, imgsz=args["imgsz"], device=dev, plots=False)
        else:
            model.train(**args)
        out = RUNS / p["name"] / "weights"
        for w in ("best", "last"):
            if (out / f"{w}.pt").exists(): shutil.copy(out / f"{w}.pt", WTS / f"{p['name']}_{w}.pt")
        log("Weights saved to library: " + p["name"] + "_best.pt / _last.pt")
        ST["state"] = "cancelled" if ST["state"] == "cancelling" else "done"
    except Exception as e:
        ST.update(state="error", error=str(e)); log(f"ERROR: {e}")

@app.post("/api/train")
def train(p: dict):
    if ST["state"] in ("running", "cancelling"): raise HTTPException(409, "A training job is already running.")
    p["name"] = "".join(c for c in p.get("name", "run") if c.isalnum() or c in "-_") or "run"
    ST.update(state="running", epoch=0, epochs=int(p["epochs"]), log=[], run=p["name"], error=None)
    threading.Thread(target=worker, args=(p,), daemon=True).start(); return {"ok": True}

@app.post("/api/train/cancel")
def cancel():
    if ST["state"] == "running":
        ST["state"] = "cancelling"; log("Cancelling after the current epoch...")
        if TRAINER["t"]: TRAINER["t"].stop = True
    return {"ok": True}

@app.get("/api/train/status")
def status(): return ST

# ---------- Module 3: evaluation ----------
@app.get("/api/runs")
def runs(): return sorted(p.name for p in RUNS.iterdir() if p.is_dir())

@app.get("/api/runs/{name}")
def run_info(name: str):
    d, m = RUNS / safe(name), {}
    f = d / "results.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open()))
        if rows:
            r = {k.strip(): v for k, v in rows[-1].items()}
            m = {"precision": r.get("metrics/precision(B)"), "recall": r.get("metrics/recall(B)"),
                 "map50": r.get("metrics/mAP50(B)"), "map5095": r.get("metrics/mAP50-95(B)")}
    plots = [p.name for p in d.glob("*.png") if p.stem.startswith(("confusion", "results", "PR_", "F1_"))]
    return {"metrics": m, "plots": plots}
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")

# ---------- Module 4: inference (strict OpenCV rendering) ----------
MODELS = {}
@app.post("/api/infer")
async def infer(file: UploadFile = File(...), weights: str = Form(...), conf: float = Form(0.25),
                unc_max: float = Form(0.5), iou: float = Form(0.45)):
    from ultralytics import YOLO
    wp = WTS / safe(weights)
    if not wp.exists(): raise HTTPException(404, "Weights not found")
    if weights not in MODELS: MODELS[weights] = YOLO(str(wp))
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Invalid image")
    res = MODELS[weights].predict(img, conf=conf, iou=iou, agnostic_nms=True, verbose=False)[0]
    h, w = img.shape[:2]; th = 1 if max(h, w) < 1400 else 2
    fs = max(0.3, min(0.55, max(h, w) / 2400)); cnt = {"Empty": 0, "Occupied": 0, "Uncertain": 0}
    overlay = img.copy(); labels = []
    for b in res.boxes:
        c, s = int(b.cls), float(b.conf); x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        k = 2 if (c == 2 or s <= unc_max) else c
        cnt[NAMES[k]] += 1; col = BGR[k]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        txt = f"{NAMES[k]} {int(s * 100)}%"
        (tw, tH), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        bw, bh = min(tw + 6, max(10, x2 - x1 - 2)), tH + 6
        if bh > (y2 - y1): continue
        labels.append((x1 + 1, y1 + 1, bw, bh, col, txt))
    for x, y, bw, bh, col, _ in labels: cv2.rectangle(overlay, (x, y), (x + bw, y + bh), col, -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for x, y, bw, bh, col, txt in labels:
        cv2.putText(img, txt, (x + 3, y + bh - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", img)  # lossless
    return {"image": "data:image/png;base64," + base64.b64encode(png).decode(),
            "total": sum(cnt.values()), **{k.lower(): v for k, v in cnt.items()}}

@app.get("/")
def index(): return FileResponse("index.html")
