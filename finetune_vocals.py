"""Fine-tune pretrained HTDemucs on MoisesDB (vocals / drums / bass / other).
Kaggle:  !pip -q install demucs soundfile   (turn Internet ON to fetch pretrained weights)
         !python finetune_vocals.py
Output:  /kaggle/working/htdemucs_vocals_ft.pt  (only saved if it beats the pretrained baseline on held-out songs)
"""
import os, sys, time, random, math, json
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from demucs.pretrained import get_model
from demucs.apply import apply_model

ROOT = Path("/kaggle/input/moisesdb/moisesdb/moisesdb_v0.1")
OUT = Path("/kaggle/working")
SR, SRC = 44100, ["drums", "bass", "other", "vocals"]      # HTDemucs source order
GROUP = {"vocals": "vocals", "drums": "drums", "percussion": "drums", "bass": "bass"}  # rest -> "other"
EPOCHS, STEPS, BS, ACCUM, LR, N_VAL, AMP = 12, 60, 4, 2, 3e-5, 10, True   # lower BS if OOM; AMP=False if loss is NaN
LOSS_W = torch.tensor([1., 1., 1., 3.])                     # care most about vocals
SMOKE = os.environ.get("SMOKE") == "1"                      # SMOKE=1 -> tiny 1-epoch test run to check everything works
if SMOKE:
    EPOCHS, STEPS, N_VAL = 1, 3, 2
dev = "cuda"
random.seed(0); torch.manual_seed(0)


LOG = open(OUT / "progress.log", "a", buffering=1)          # line-buffered: readable while running


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOG.write(line + "\n")


def scan():
    """Group every wav under its stem class. A MoisesDB folder can hold SEVERAL wavs (lead, backing, ...) -> sum them all."""
    songs = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        g = {k: [] for k in SRC}
        for sub in d.iterdir():
            if sub.is_dir():
                g[GROUP.get(sub.name.lower(), "other")] += [str(p) for p in sorted(sub.glob("*.wav"))]
        if g["vocals"]:
            g["n"] = min(sf.info(p).frames for k in SRC for p in g[k])
            songs.append(g)
    return songs


def read(path, start, n):
    x, _ = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
    x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]
    return np.pad(x.T, ((0, 0), (0, n - x.shape[0])))


def stem(song, k, start, n):
    return sum((read(p, start, n) for p in song[k]), np.zeros((2, n), np.float32))


class DS(Dataset):
    def __init__(s, songs, n, seg):
        s.songs, s.n, s.seg = songs, n, seg

    def __len__(s):
        return s.n

    def __getitem__(s, i):
        coherent = random.random() < 0.5                    # 50%: real mix, 50%: stems from different songs (remix aug)
        song = random.choice(s.songs); start = random.randint(0, max(0, song["n"] - s.seg))
        out = []
        for k in SRC:
            for _ in range(5):                              # avoid all-silent vocal crops
                if not coherent:
                    song = random.choice(s.songs); start = random.randint(0, max(0, song["n"] - s.seg))
                x = stem(song, k, start, s.seg)
                if k != "vocals" or x.std() > 0.01:
                    break
            x = x * random.uniform(0.5, 1.5)
            x = -x if random.random() < 0.5 else x
            x = x[::-1] if random.random() < 0.5 else x    # channel swap
            out.append(torch.from_numpy(np.ascontiguousarray(x)))
        return torch.stack(out)                             # (4, 2, T)


def sdr(ref, est, eps=1e-8):
    return 10 * torch.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + eps) + eps).item()


def make_val(songs, secs=30):
    """Fixed 30 s excerpts (the most vocal-heavy of 6 random windows) from held-out songs."""
    n, val = secs * SR, []
    for s in songs:
        starts = [random.randint(0, max(0, s["n"] - n)) for _ in range(6)]
        st = max(starts, key=lambda a: stem(s, "vocals", a, n).std())
        parts = {k: torch.from_numpy(stem(s, k, st, n)) for k in SRC}
        val.append((sum(parts.values()), parts["vocals"]))
    return val


@torch.no_grad()
def evaluate(model, val):
    model.eval(); scores = []
    for mix, voc in val:
        ref = mix.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8
        est = apply_model(model, ((mix - mu) / sd)[None].to(dev), device=dev, split=True, overlap=0.25, progress=False)[0]
        scores.append(sdr(voc, (est[3].cpu() * sd + mu)))
    return float(np.mean(scores))


def load_base():
    m = get_model("htdemucs")
    m = m.models[0] if hasattr(m, "models") else m
    assert list(m.sources) == SRC
    return m


if __name__ == "__main__":
    t0 = time.time()
    log(f"start | SMOKE={SMOKE} | GPU={torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    songs = scan(); random.shuffle(songs)
    val_songs, train_songs = songs[:N_VAL], songs[N_VAL:]
    log(f"{len(train_songs)} train songs, {len(val_songs)} val songs")
    val = make_val(val_songs)
    log("validation excerpts built")

    model = load_base().to(dev)
    base = evaluate(model, val); best = base
    log(f"Pretrained baseline vocal SDR on held-out songs: {base:.2f} dB")

    seg = int(model.segment * SR)                           # 7.8 s, what HTDemucs was built for
    total = EPOCHS * STEPS
    dl = DataLoader(DS(train_songs, total * ACCUM * BS, seg), batch_size=BS, num_workers=4, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    it, hist = iter(dl), [{"epoch": 0, "val_sdr": base}]

    for ep in range(1, EPOCHS + 1):
        model.train(); run = 0.0
        for step in range(1, STEPS + 1):
            for _ in range(ACCUM):
                y = next(it).to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    est = model(y.sum(1))
                per_src = (est.float() - y).abs().mean((0, 2, 3))
                loss = (per_src * LOSS_W.to(dev)).sum() / LOSS_W.sum() / ACCUM
                scaler.scale(loss).backward(); run += loss.item()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if step == 1 or step % 10 == 0:
                log(f"  epoch {ep} step {step}/{STEPS} | loss {run / step:.4f} | {(time.time() - t0) / 60:.1f} min elapsed")
        score = evaluate(model, val)
        hist.append({"epoch": ep, "train_loss": run / STEPS, "val_sdr": score})
        flag = ""
        if score > best:
            best = score; torch.save(model.state_dict(), OUT / "htdemucs_vocals_ft.pt"); flag = " <- saved"
        log(f"epoch {ep:2d} | train L1 {run / STEPS:.4f} | val vocal SDR {score:.2f} dB (baseline {base:.2f}){flag}")
        json.dump(hist, open(OUT / "finetune_history.json", "w"), indent=1)

    log(f"Done in {(time.time() - t0) / 60:.1f} min. baseline {base:.2f} dB -> best {best:.2f} dB. "
        + ("Fine-tuned weights saved." if best > base else "No gain: keep using the pretrained model."))
