"""Run:  uvicorn server:app --host 0.0.0.0 --port 8000      (needs ffmpeg for mp3/m4a)
Uses pretrained HTDemucs, plus your fine-tuned weights if htdemucs_vocals_ft.pt sits next to this file (or set FT_WEIGHTS).
"""
import os, shutil, threading, uuid
from pathlib import Path
import librosa, numpy as np, soundfile as sf, torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 44100
HERE = Path(__file__).parent
JOBS = HERE / "jobs"; JOBS.mkdir(exist_ok=True)
WEIGHTS = Path(os.environ.get("FT_WEIGHTS", HERE / "htdemucs_vocals_ft.pt"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model("htdemucs")
model = model.models[0] if hasattr(model, "models") else model
if WEIGHTS.exists():
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    print("Loaded fine-tuned weights:", WEIGHTS)
else:
    print("Using pretrained HTDemucs (no fine-tuned weights found)")
model.to(dev).eval()
lock = threading.Lock()                                    # one separation at a time on the GPU
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.post("/separate")
def separate(file: UploadFile = File(...)):                # sync def -> runs in a worker thread
    job = uuid.uuid4().hex[:12]; d = JOBS / job; d.mkdir()
    src = d / ("input" + Path(file.filename or "x.wav").suffix)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        y, _ = librosa.load(src, sr=SR, mono=False)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not decode audio: {e}")
    src.unlink(missing_ok=True)

    y = torch.from_numpy(np.atleast_2d(y)).float()
    y = y.repeat(2, 1) if y.shape[0] == 1 else y[:2]
    ref = y.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8   # same normalisation demucs uses
    with lock, torch.no_grad():
        est = apply_model(model, ((y - mu) / sd)[None].to(dev), device=dev, split=True, overlap=0.25, shifts=1)[0]
    est = est.cpu() * sd + mu                              # (4, 2, T): drums, bass, other, vocals
    vocals, inst = est[3], est[:3].sum(0)
    peak = max(vocals.abs().max().item(), inst.abs().max().item(), 1.0)
    for name, wav in [("vocals", vocals), ("instrumental", inst)]:
        sf.write(d / f"{name}.wav", (wav / peak).T.numpy(), SR, subtype="PCM_16")
    return {"vocals": f"/files/{job}/vocals.wav", "instrumental": f"/files/{job}/instrumental.wav"}


@app.get("/files/{job}/{name}")
def files(job: str, name: str):
    if not job.isalnum() or name not in ("vocals.wav", "instrumental.wav"):
        raise HTTPException(404)
    p = JOBS / job / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav", filename=name)
