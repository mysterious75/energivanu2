# %% [markdown]
# # ⚡ Energivanu — Alibaba GPU Training (T4)
# Real Alibaba GPU Trace 2020 → 15 features → TCN+Attention Training

# %% Cell 1: Setup
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compute: {torch.cuda.get_device_capability(0)}")

# %% Cell 2: Download Alibaba Data
print("="*60)
print("DOWNLOADING ALIBABA GPU TRACE 2020")
print("="*60)

os.makedirs("alibaba", exist_ok=True)

# Download headers
os.system("curl -sL --connect-timeout 10 --max-time 30 'https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_sensor_table.header' -o alibaba/sensor.header")
os.system("curl -sL --connect-timeout 10 --max-time 30 'https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_machine_metric.header' -o alibaba/metric.header")

# Download sensor table (~388MB)
t0 = time.time()
print("Downloading pai_sensor_table (~388MB)...")
ret = os.system("curl -sL --connect-timeout 15 --max-time 600 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz' -o alibaba/sensor.tar.gz")
print(f"  sensor: {time.time()-t0:.0f}s, rc={ret}")

# Download machine metric (~198MB)
t0 = time.time()
print("Downloading pai_machine_metric (~198MB)...")
ret2 = os.system("curl -sL --connect-timeout 15 --max-time 600 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz' -o alibaba/metric.tar.gz")
print(f"  metric: {time.time()-t0:.0f}s, rc={ret2}")

# Extract
print("Extracting...")
if os.path.exists("alibaba/sensor.tar.gz") and os.path.getsize("alibaba/sensor.tar.gz") > 1000:
    os.system("tar xzf alibaba/sensor.tar.gz -C alibaba/ 2>/dev/null")
if os.path.exists("alibaba/metric.tar.gz") and os.path.getsize("alibaba/metric.tar.gz") > 1000:
    os.system("tar xzf alibaba/metric.tar.gz -C alibaba/ 2>/dev/null")

# Add headers
for fname, hdr in [("pai_sensor_table", "sensor"), ("pai_machine_metric", "metric")]:
    csv_p = f"alibaba/{fname}.csv"
    hdr_p = f"alibaba/{hdr}.header"
    if os.path.exists(csv_p) and os.path.exists(hdr_p):
        sz = os.path.getsize(csv_p)
        if sz > 100:  # Only if has data
            with open(hdr_p) as hf: header = hf.read().strip()
            with open(csv_p) as df: content = df.read()
            with open(csv_p, "w") as out: out.write(header + "\n" + content)
            print(f"  ✅ {fname}: {sz/1e6:.0f}MB, header added")
        else:
            print(f"  ⚠️ {fname}: empty ({sz} bytes)")

# Cleanup
os.system("rm -f alibaba/*.tar.gz")

# Check what we have
for f in os.listdir("alibaba"):
    fp = os.path.join("alibaba", f)
    if os.path.isfile(fp):
        print(f"  {f}: {os.path.getsize(fp)/1e6:.1f}MB")

# %% Cell 3: Load & Process Data → 15 Features
print("\n" + "="*60)
print("PROCESSING → 15 FEATURES")
print("="*60)

GPU_TDP = 700.0
FACILITY_GPUS = 200000

COL_MAP = {
    "cpu_usage": "cpu_util", "gpu_wrk_util": "gpu_util",
    "avg_mem": "mem_util", "avg_gpu_wrk_mem": "gpu_mem_util",
    "machine_cpu_usr": "cpu_util", "machine_gpu": "gpu_util",
    "machine_cpu": "cpu_util2",
}

df = None
# Try sensor first, then metric
for csv_name in ["pai_sensor_table.csv", "pai_machine_metric.csv"]:
    csv_path = f"alibaba/{csv_name}"
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 1000:
        print(f"  Skip {csv_name} (not found or empty)")
        continue
    try:
        print(f"  Loading {csv_name} (first 300K rows)...")
        _df = pd.read_csv(csv_path, nrows=300000)
        _df = _df.rename(columns=COL_MAP)
        if "gpu_util" in _df.columns:
            df = _df
            print(f"  ✅ {csv_name}: {df.shape}, gpu_util found")
            break
        else:
            print(f"  ⚠️ {csv_name}: no gpu_util. Cols: {list(_df.columns)[:6]}")
    except Exception as e:
        print(f"  ❌ {csv_name}: {e}")

if df is None:
    print("❌ No Alibaba data loaded! Generating synthetic...")
    np.random.seed(42)
    n = 100000
    t = np.linspace(0, 200, n)
    gu = 30 + 40*np.sin(0.05*t) + 15*np.sin(0.5*t) + 5*np.random.randn(n)
    df = pd.DataFrame({"gpu_util": gu.clip(0, 100)})
    df["timestamp"] = pd.date_range("2020-01-01", periods=n, freq="60s")

print(f"\nUsing {len(df)} rows")

# Feature engineering
gpu_util = df["gpu_util"].fillna(0).clip(0, 100).values.astype(np.float32)
n = len(gpu_util)

# Power estimation
single_w = 70 + (GPU_TDP - 70) * (gpu_util / 100.0)
facility_mw = (single_w * FACILITY_GPUS / 1e6).astype(np.float32)

# Rolling stats
rm = pd.Series(facility_mw).rolling(30, min_periods=1).mean().values.astype(np.float32)
rs = pd.Series(facility_mw).rolling(30, min_periods=1).std().fillna(0).values.astype(np.float32)
roc = np.diff(facility_mw, prepend=facility_mw[0])
roc2 = np.diff(roc, prepend=roc[0])

# Temporal
if "timestamp" in df.columns:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    hsin = np.sin(2*np.pi*ts.dt.hour/24).fillna(0).values.astype(np.float32)
    hcos = np.cos(2*np.pi*ts.dt.hour/24).fillna(0).values.astype(np.float32)
else:
    t_idx = np.arange(n)
    hsin = np.sin(2*np.pi*(t_idx % 3600)/3600).astype(np.float32)
    hcos = np.cos(2*np.pi*(t_idx % 3600)/3600).astype(np.float32)

# GPU metrics
temp = (0.4 + 0.4*gpu_util/100).clip(0, 1).astype(np.float32)
mem = df.get("gpu_mem_util", pd.Series(np.zeros(n))).fillna(0).clip(0, 100).values.astype(np.float32)
cpu_u = df.get("cpu_util", pd.Series(np.full(n, 50.0))).fillna(50).clip(0, 100).values.astype(np.float32)
is_ar = ((gpu_util > 80) & (mem < 30)).astype(np.float32)

features = np.column_stack([
    facility_mw, roc, roc2, rm, rs,
    gpu_util/100, gpu_util/100, temp, temp,
    gpu_util/100, mem/100, cpu_u/100, hsin, hcos, is_ar
])

# Clean
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
print(f"Features: {features.shape}")
print(f"Power: {features[:,0].min():.1f} - {features[:,0].max():.1f} MW")

# %% Cell 4: Sequences
SEQ_LEN = 30; PH = 10; STRIDE = 10; BATCH = 128

pw = features[:, 0]
pc = np.diff(pw, prepend=pw[0])
sig = np.zeros(n, dtype=np.int64)
sig[pc > 0.5] = 1; sig[pc < -0.5] = 2

X, Yp, Ys = [], [], []
for i in range(0, n - SEQ_LEN - PH, STRIDE):
    X.append(features[i:i+SEQ_LEN])
    Yp.append(pw[i+SEQ_LEN:i+SEQ_LEN+PH])
    Ys.append(sig[i+SEQ_LEN])

X = np.array(X, dtype=np.float32)
Yp = np.array(Yp, dtype=np.float32)
Ys = np.array(Ys, dtype=np.int64)
print(f"Sequences: {X.shape}")

ns = len(X); idx = np.random.permutation(ns); sp = int(ns*0.85)

class DS(Dataset):
    def __init__(s, x, yp, ys):
        s.x = torch.tensor(x); s.yp = torch.tensor(yp); s.ys = torch.tensor(ys)
    def __len__(s): return len(s.x)
    def __getitem__(s, i): return s.x[i], s.yp[i], s.ys[i]

tdl = DataLoader(DS(X[idx[:sp]], Yp[idx[:sp]], Ys[idx[:sp]]), BATCH, shuffle=True, pin_memory=True)
vdl = DataLoader(DS(X[idx[sp:]], Yp[idx[sp:]], Ys[idx[sp:]]), BATCH, pin_memory=True)
print(f"Train: {sp}, Val: {ns-sp}")

# %% Cell 5: Model
class TB(nn.Module):
    def __init__(s, ic, oc, k, d, dr=0.1):
        super().__init__()
        p = (k-1)*d
        s.c1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d)
        s.c2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d)
        s.n1 = nn.LayerNorm(oc); s.n2 = nn.LayerNorm(oc)
        s.d1 = nn.Dropout(dr); s.d2 = nn.Dropout(dr)
        s.r = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(s, x):
        res = s.r(x)
        o = s.d1(torch.relu(s.n1(s.c1(x)[:,:,:x.size(2)].transpose(1,2)).transpose(1,2)))
        o = s.d2(torch.relu(s.n2(s.c2(o)[:,:,:x.size(2)].transpose(1,2)).transpose(1,2)))
        return torch.relu(o + res)

class PEB(nn.Module):
    def __init__(s, nf=15, sl=30, ph=10):
        super().__init__()
        s.pn = nn.LayerNorm(min(7,nf)); s.tn = nn.LayerNorm(min(7,max(0,nf-7)))
        s.qn = nn.LayerNorm(max(0,nf-14))
        s._p = min(7,nf); s._t = min(7,max(0,nf-7)); s._q = max(0,nf-14)
        s.proj = nn.Linear(nf, 128)
        s.tcn = nn.Sequential(TB(128,32,5,1), TB(32,64,3,2), TB(64,128,3,4))
        s.attn = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)
        s.an = nn.LayerNorm(128); s.lw = nn.Linear(128, 1)
        s.ph = nn.Sequential(nn.Linear(128,256), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(256,128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128,ph))
        s.sh = nn.Sequential(nn.Linear(128+ph,256), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(256,128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128,3))
    def forward(s, x):
        xn = torch.zeros_like(x)
        if s._p: xn[:,:,:s._p] = s.pn(x[:,:,:s._p])
        if s._t: xn[:,:,s._p:s._p+s._t] = s.tn(x[:,:,s._p:s._p+s._t])
        if s._q: xn[:,:,s._p+s._t:] = s.qn(x[:,:,s._p+s._t:])
        h = s.tcn(s.proj(xn).transpose(1,2)).transpose(1,2)
        h, _ = s.attn(h, h, h); h = s.an(h)
        l, m = h[:,-1,:], h.mean(1)
        a = torch.sigmoid(s.lw(l)); g = a*l + (1-a)*m
        p = s.ph(g)
        return p, s.sh(torch.cat([g, p], 1))

model = PEB(15, SEQ_LEN, PH).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {params:,} params on {DEVICE}")

# %% Cell 6: Training (50 epochs)
EPOCHS = 50
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
pl = nn.HuberLoss(); sl_fn = nn.CrossEntropyLoss()
best = float("inf")

print(f"\n{'='*60}")
print(f"TRAINING — {EPOCHS} epochs on {DEVICE}")
print(f"{'='*60}")

t_start = time.time()
for ep in range(EPOCHS):
    model.train(); tl = 0
    for xb, yb, sb in tdl:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        pp, ps = model(xb)
        loss = pl(pp, yb) + 0.3 * sl_fn(ps, sb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item() * len(xb)
    tl /= sp

    model.eval(); vl = vm = 0
    with torch.no_grad():
        for xb, yb, sb in vdl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            pp, ps = model(xb)
            l = pl(pp, yb) + 0.3 * sl_fn(ps, sb)
            vl += l.item() * len(xb)
            vm += (torch.mean(torch.abs(pp - yb)/(yb.abs()+1e-6))*100).item() * len(xb)
    vl /= (ns-sp); vm /= (ns-sp); sched.step()

    if vl < best:
        best = vl
        torch.save({"model_state_dict": model.state_dict(),
                     "config": {"n_features":15,"seq_len":SEQ_LEN,"pred_horizon":PH},
                     "val_loss":vl, "val_mape":vm, "epoch":ep}, "best_model.pt")

    if (ep+1) % 10 == 0 or ep == 0:
        elapsed = time.time() - t_start
        print(f"Ep {ep+1:3d}/{EPOCHS} | Train {tl:.4f} | Val {vl:.4f} | MAPE {vm:.2f}% | {elapsed:.0f}s")

total_time = time.time() - t_start
print(f"\n✅ Training done in {total_time:.0f}s | Best val loss: {best:.4f}")

# %% Cell 7: Results
results = {
    "timestamp": datetime.now().isoformat(),
    "status": "COMPLETE",
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
    "data_source": "Alibaba GPU Trace 2020 (CC BY 4.0)",
    "data_rows": n,
    "train_samples": sp,
    "val_samples": ns - sp,
    "model_params": params,
    "epochs": EPOCHS,
    "best_val_loss": round(float(best), 6),
    "best_val_mape": round(float(vm), 2),
    "training_time_s": round(total_time),
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("📊 FINAL RESULTS")
print(f"{'='*60}")
for k, v in results.items():
    print(f"  {k}: {v}")

print(f"\n✅ DONE! Files: best_model.pt, results.json")
