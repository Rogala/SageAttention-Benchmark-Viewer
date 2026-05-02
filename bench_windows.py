"""
SageAttention Model Benchmark — ComfyUI Models

All attention parameters verified with Attention Logger (rogala) in ComfyUI.
Workflow: Load Model -> [Attention Logger rogala] -> KSampler

Kernel sources are printed at startup for full transparency.

Usage:
    python sage_bench_models.py
    python sage_bench_models.py --warmup 20 --iters 100

Output file is named automatically from GPU model and VRAM, e.g.:
    5060-ti-16.json
    4070-ti_super-16.json
    3060-laptop-6.json
"""

import json, time, sys, platform, argparse, statistics, subprocess, inspect, re, threading, math
from pathlib import Path

import torch
import torch.nn.functional as F

# ── imports ───────────────────────────────────────────────────────────────────
try:
    from sageattention import sageattn
    HAS_SA2 = True
    SA2_FILE = inspect.getfile(sageattn)
except ImportError:
    HAS_SA2 = False
    SA2_FILE = "NOT FOUND"

try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    HAS_SA2_FP8 = True
    SA2_FP8_FILE = inspect.getfile(sageattn_qk_int8_pv_fp8_cuda)
except ImportError:
    HAS_SA2_FP8 = False
    SA2_FP8_FILE = "NOT FOUND"

# Try importing SA3 alongside SA2 — works on newer versions
# Falls back to subprocess if conflict detected
SA3_SUBPROCESS = False
try:
    from sageattn3.api import (blockscaled_fp4_attn, preprocess_qkv,
                                scale_and_quant_fp4, scale_and_quant_fp4_transpose,
                                scale_and_quant_fp4_permute)
    # verify no conflict by doing a quick test import together with SA2
    HAS_SA3 = True
    SA3_FILE = inspect.getfile(blockscaled_fp4_attn)
    SA3_SUBPROCESS = False
except Exception:
    # conflict or not installed — try subprocess mode
    try:
        r = __import__('subprocess').run(
            [__import__('sys').executable, '-c',
             'from sageattn3.api import blockscaled_fp4_attn; print("ok")'],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and 'ok' in r.stdout:
            HAS_SA3 = True
            SA3_FILE = "via subprocess (conflict with SA2 in same process)"
            SA3_SUBPROCESS = True
        else:
            HAS_SA3 = False
            SA3_FILE = "NOT FOUND"
    except Exception:
        HAS_SA3 = False
        SA3_FILE = "NOT FOUND"

# ── GPU output filename ───────────────────────────────────────────────────────
def gpu_filename():
    """Generate output filename from GPU name and VRAM.
    Examples:
        NVIDIA GeForce RTX 5060 Ti  -> 5060-ti-16.json
        NVIDIA GeForce RTX 4070 Ti SUPER -> 4070-ti_super-16.json
        NVIDIA GeForce RTX 3060 Laptop GPU -> 3060-laptop-6.json
    """
    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    vram = 2 ** round(math.log2(total_gib))

    name = re.sub(r"NVIDIA GeForce RTX\s*", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"NVIDIA GeForce\s*",      "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"NVIDIA\s*",              "", name, flags=re.IGNORECASE).strip()
    name = name.replace("Ti SUPER", "ti_super")
    name = re.sub(r"Laptop GPU", "laptop", name, flags=re.IGNORECASE)
    name = re.sub(r"\bGPU\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", "-", name.strip()).lower()
    name = re.sub(r"-+", "-", name).strip("-")

    return f"{name}-{vram}.json"

# ── model configs ─────────────────────────────────────────────────────────────
# Format: (batch, heads, seq, head_dim, label, dtype)
# All params verified with Attention Logger (rogala) in ComfyUI.
# One run per resolution is sufficient — SA mode does not affect tensor shapes.

CONFIGS = [

    # ── Image: SDXL-1.0 ──────────────────────────────────────────────────────
    # verified: 1024x1024 / 1344x768
    (1, 10, 4096,  64, "[SDXL-1.0] main-attn 1024x1024 fp16",        torch.float16),
    (1, 20, 1024,  64, "[SDXL-1.0] mid-attn  1024x1024 fp16",        torch.float16),
    (1, 10, 4032,  64, "[SDXL-1.0] main-attn 1344x768  fp16",        torch.float16),
    (1, 20, 1008,  64, "[SDXL-1.0] mid-attn  1344x768  fp16",        torch.float16),

    # ── Image: SD3.5-Large ────────────────────────────────────────────────────
    # verified: 1024x1024 / 1344x768 / 1152x1152 / 1536x832
    (1, 38, 4250,  64, "[SD3.5-Large] 1024x1024 fp16",                torch.float16),
    (1, 38, 4186,  64, "[SD3.5-Large] 1344x768  fp16",                torch.float16),
    (1, 38, 5338,  64, "[SD3.5-Large] 1152x1152 fp16",                torch.float16),
    (1, 38, 5146,  64, "[SD3.5-Large] 1536x832  fp16",                torch.float16),

    # ── Image: Flux.1-Dev (Kontext, Krea) ────────────────────────────────────
    # verified: 1024x1024 / 1344x768 / 1216x1216 / 1664x896
    (1, 24, 4352, 128, "[Flux.1-Dev (Kontext, Krea)] 1024x1024 bf16", torch.bfloat16),
    (1, 24, 4288, 128, "[Flux.1-Dev (Kontext, Krea)] 1344x768  bf16", torch.bfloat16),
    (1, 24, 6032, 128, "[Flux.1-Dev (Kontext, Krea)] 1216x1216 bf16", torch.bfloat16),
    (1, 24, 6080, 128, "[Flux.1-Dev (Kontext, Krea)] 1664x896  bf16", torch.bfloat16),

    # ── Image: Flux.2-Dev ─────────────────────────────────────────────────────
    # verified: 1024x1024 / 1344x768 / 1440x1440 / 1920x1088
    (1, 48, 4608, 128, "[Flux.2-Dev] 1024x1024 bf16",                 torch.bfloat16),
    (1, 48, 4544, 128, "[Flux.2-Dev] 1344x768  bf16",                 torch.bfloat16),
    (1, 48, 8612, 128, "[Flux.2-Dev] 1440x1440 bf16",                 torch.bfloat16),
    (1, 48, 8672, 128, "[Flux.2-Dev] 1920x1088 bf16",                 torch.bfloat16),

    # ── Image: Flux.2-Dev Klein 9B ────────────────────────────────────────────
    # verified: 1024x1024 / 1344x768 / 1440x1440 / 1920x1088
    (1, 32, 4356, 128, "[Flux.2-Dev (Klein 9B)] 1024x1024 bf16",      torch.bfloat16),
    (1, 32, 4413, 128, "[Flux.2-Dev (Klein 9B)] 1344x768  bf16",      torch.bfloat16),
    (1, 32, 8793, 128, "[Flux.2-Dev (Klein 9B)] 1440x1440 bf16",      torch.bfloat16),
    (1, 32, 8740, 128, "[Flux.2-Dev (Klein 9B)] 1920x1088 bf16",      torch.bfloat16),

    # ── Image: Z-Image / Z-Image Turbo ───────────────────────────────────────
    # verified: 1024x1024 / 1344x768 / 1280x1280 / 1600x896
    # joint attention: img tokens + 32 txt tokens concatenated
    (1, 30, 4128, 128, "[Z-Image (Turbo)] 1024x1024 bf16",            torch.bfloat16),
    (1, 30, 4064, 128, "[Z-Image (Turbo)] 1344x768  bf16",            torch.bfloat16),
    (1, 30, 6432, 128, "[Z-Image (Turbo)] 1280x1280 bf16",            torch.bfloat16),
    (1, 30, 5632, 128, "[Z-Image (Turbo)] 1600x896  bf16",            torch.bfloat16),

    # ── Image: Qwen-Image-2512 ────────────────────────────────────────────────
    # verified: 1328x1328 / 1664x928 / 1920x1920 / 2560x1440
    (1, 24,  6978, 128, "[Qwen-Image-2512] 1328x1328 bf16",           torch.bfloat16),
    (1, 24,  6121, 128, "[Qwen-Image-2512] 1664x928  bf16",           torch.bfloat16),
    (1, 24, 14489, 128, "[Qwen-Image-2512] 1920x1920 bf16",           torch.bfloat16),
    (1, 24, 14489, 128, "[Qwen-Image-2512] 2560x1440 bf16",           torch.bfloat16),

    # ── Image: Qwen-Image-Edit-2511 ───────────────────────────────────────────
    # verified: 1328x1328 / 1664x928 / 1920x1920 / 2560x1440
    # two attention layers per step — largest seq taken per resolution
    (1, 24,  6912, 128, "[Qwen-Image-Edit-2511] 1328x1328 bf16",      torch.bfloat16),
    (1, 24,  6055, 128, "[Qwen-Image-Edit-2511] 1664x928  bf16",      torch.bfloat16),
    (1, 24, 14423, 128, "[Qwen-Image-Edit-2511] 1920x1920 bf16",      torch.bfloat16),
    (1, 24, 14423, 128, "[Qwen-Image-Edit-2511] 2560x1440 bf16",      torch.bfloat16),

    # ── Image: ERNIE-Image / ERNIE-Image Turbo ───────────────────────────────
    # verified via global patch (patch_global=True): 1024x1024 / 1344x768
    (1, 32, 4097, 128, "[ERNIE-Image (Turbo)] 1024x1024 bf16",        torch.bfloat16),
    (1, 32, 4033, 128, "[ERNIE-Image (Turbo)] 1344x768  bf16",        torch.bfloat16),

    # ── Video: LTX-2.3 ───────────────────────────────────────────────────────
    # verified: 960x544 / 1280x736 / 1600x960 / 1920x1088
    # hd=128 main self-attention only; hd=64 cross-attn excluded
    (1, 32,  8160, 128, "[LTX-2.3] 960x544   bf16",                   torch.bfloat16),
    (1, 32, 14720, 128, "[LTX-2.3] 1280x736  bf16",                   torch.bfloat16),
    (1, 32, 24000, 128, "[LTX-2.3] 1600x960  bf16",                   torch.bfloat16),
    (1, 32, 32640, 128, "[LTX-2.3] 1920x1088 bf16",                   torch.bfloat16),

    # ── Video: Wan2.2 ─────────────────────────────────────────────────────────
    # verified: 832x480 / 960x544 / 1280x720 / 1920x1088
    # two passes per step, sequential — same seq but first pass uses ~half the denoising steps
    # real total load ~1.5x single pass; benchmark times one pass (worst case = second pass)
    (1, 40,  7800, 128, "[Wan2.2] 832x480   fp16",                    torch.float16),
    (1, 40, 10200, 128, "[Wan2.2] 960x544   fp16",                    torch.float16),
    (1, 40, 18000, 128, "[Wan2.2] 1280x720  fp16",                    torch.float16),
    (1, 40, 40800, 128, "[Wan2.2] 1920x1088 fp16",                    torch.float16),

    # ── Video: HunyuanVideo-1.5 ───────────────────────────────────────────────
    # verified: 848x480 / 1280x720 / 1920x1088
    # two close seq values observed per resolution — largest taken
    (1, 16,  49367, 128, "[HunyuanVideo-1.5] 848x480   fp16",         torch.float16),
    (1, 16, 111677, 128, "[HunyuanVideo-1.5] 1280x720  fp16",         torch.float16),
    (1, 16, 253037, 128, "[HunyuanVideo-1.5] 1920x1088 fp16",         torch.float16),

    # ── Audio: ACE-Step-1.5 ───────────────────────────────────────────────────
    # verified: 50s / 100s / 150s audio generation
    # seq=326 is fixed text/lyrics encoder; audio seq scales with duration
    (1, 16,   326, 128, "[ACE-Step-1.5] text-attn        bf16",       torch.bfloat16),
    (1, 16,  1250, 128, "[ACE-Step-1.5] audio 50s        bf16",       torch.bfloat16),
    (1, 32,   625, 128, "[ACE-Step-1.5] audio 50s cross  bf16",       torch.bfloat16),
    (1, 16,  2500, 128, "[ACE-Step-1.5] audio 100s       bf16",       torch.bfloat16),
    (1, 32,  1250, 128, "[ACE-Step-1.5] audio 100s cross bf16",       torch.bfloat16),
    (1, 16,  3750, 128, "[ACE-Step-1.5] audio 150s       bf16",       torch.bfloat16),
    (1, 32,  1875, 128, "[ACE-Step-1.5] audio 150s cross bf16",       torch.bfloat16),
]

# ── VRAM minimums — based on real measurements (RTX 5060 Ti 16GB) ────────────
# SA3 peak includes FP4 quantization buffers (preprocess_qkv overhead)
# All other configs pass dynamic vram_available_gb() check automatically
VRAM_LIMITS = {
    # LTX — SA3 peak 4.2 GB
    "[LTX-2.3] 1920x1088 bf16":            6,
    # Wan2.2 — SA3 peaks: 3.2 GB / 8.2 GB
    "[Wan2.2] 1280x720  fp16":              4,
    "[Wan2.2] 1920x1088 fp16":             10,
    # HunyuanVideo — two passes simultaneously + SA3 buffers
    # SA3 peaks: 4.5 GB / 14.6 GB
    "[HunyuanVideo-1.5] 848x480   fp16":    6,
    "[HunyuanVideo-1.5] 1280x720  fp16":   16,
    "[HunyuanVideo-1.5] 1920x1088 fp16":   24,  # extrapolated: ~33GB expected, 24GB minimum to attempt
}

def get_min_vram_limit(label, has_sa3):
    # VRAM_LIMITS are SA3-specific (FP4 quantization buffers overhead).
    # Without SA3 the dynamic vram_available_gb() check is sufficient.
    if not has_sa3:
        return 0
    return VRAM_LIMITS.get(label, 0)


def vram_required_gb(batch, heads, seq, hd, dtype):
    """Estimate VRAM needed for Q+K+V tensors in GB.
    NOTE: only Q+K+V are counted here. Real peak is higher because of the
    output tensor and softmax workspace; the dynamic vram_available_gb()
    check provides additional headroom for that overhead.
    """
    elem = batch * heads * seq * hd
    bytes_per_elem = 2  # fp16 and bf16 both 2 bytes
    return (elem * 3 * bytes_per_elem) / 1e9

def vram_available_gb():
    """Free VRAM on device in GB (total minus PyTorch reserved pool)."""
    reserved = torch.cuda.memory_reserved(0)
    total    = torch.cuda.get_device_properties(0).total_memory
    return (total - reserved) / 1e9


# ── nvidia-smi VRAM monitor ──────────────────────────────────────────────────
class VramMonitor:
    """Polls nvidia-smi in background thread, records peak used VRAM in MiB."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                val = int(r.stdout.strip().splitlines()[0])
                if val > self._peak:
                    self._peak = val
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _poll_once(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            return int(r.stdout.strip().splitlines()[0])
        except Exception:
            return 0

    def start(self):
        self._peak = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak  # MiB


def bench(fn, *args, warmup=10, iters=50, **kwargs):
    # NOTE: VramMonitor starts AFTER warmup so reported peak reflects the
    # steady-state usage during the timed iterations, not transient warmup spikes.
    monitor = VramMonitor(interval=0.05)
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        monitor.start()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    peak_vram_mib = monitor.stop()
    return times, peak_vram_mib

def tflops(batch, heads, seq, hd, median_ms):
    """Standard attention FLOPS: 4 * B * H * S^2 * D
    NOTE: For SA3, `seq` passed in is the padded_seq the kernel actually
    operates on (not the original logical seq). This means SA3 TFLOPS reflect
    raw kernel throughput, not effective throughput on the user's data.
    """
    flops = 4 * batch * heads * (seq ** 2) * hd
    return round(flops / (median_ms / 1000) / 1e12, 3)

def stats(times, peak_vram_mib, batch, heads, seq, hd):
    med = statistics.median(times)
    return {
        "mean_ms":       round(statistics.mean(times),  4),
        "median_ms":     round(med,                     4),
        "min_ms":        round(min(times),              4),
        "max_ms":        round(max(times),              4),
        "stdev_ms":      round(statistics.stdev(times), 4),
        "peak_vram_mib": peak_vram_mib,
        "tflops":        tflops(batch, heads, seq, hd, med),
    }

# ── SA3 subprocess worker ─────────────────────────────────────────────────────
# preprocess_qkv is excluded from benchmark — only the core attention kernel is timed.
# preprocess_qkv pads seq to the nearest multiple of 128 and crops output back
# to original seq length — transparent to the caller, reported in output.
SA3_WORKER = '\nimport json,sys,time,statistics,torch,subprocess,threading\nfrom sageattn3.api import (blockscaled_fp4_attn,preprocess_qkv,\n                            scale_and_quant_fp4,scale_and_quant_fp4_transpose,\n                            scale_and_quant_fp4_permute)\n\ndef _smi_peak():\n    peak=[0]; stop=threading.Event()\n    def poll():\n        while not stop.is_set():\n            try:\n                r=subprocess.run(["nvidia-smi","--query-gpu=memory.used",\n                    "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2)\n                v=int(r.stdout.strip().splitlines()[0])\n                if v>peak[0]: peak[0]=v\n            except: pass\n            stop.wait(0.05)\n    t=threading.Thread(target=poll,daemon=True); t.start()\n    return stop,peak,t\n\ncfg=json.loads(sys.argv[1])\ndtype=torch.float16 if cfg["dtype"]=="fp16" else torch.bfloat16\nq=torch.randn(cfg["batch"],cfg["heads"],cfg["seq"],cfg["hd"],dtype=dtype,device="cuda")\nk=torch.randn(cfg["batch"],cfg["heads"],cfg["seq"],cfg["hd"],dtype=dtype,device="cuda")\nv=torch.randn(cfg["batch"],cfg["heads"],cfg["seq"],cfg["hd"],dtype=dtype,device="cuda")\n\nwith torch.no_grad():\n    qc,kc,vc,ds=preprocess_qkv(q,k,v)\n    ql=scale_and_quant_fp4(qc)\n    kl=scale_and_quant_fp4_permute(kc)\n    vl=scale_and_quant_fp4_transpose(vc)\n    KL=qc.shape[2]\n    padded_seq=qc.shape[2]\n    is_bf16=(dtype==torch.bfloat16)\n\n    def sa3_core():\n        return blockscaled_fp4_attn(ql,kl,vl,ds,KL,is_causal=False,is_bf16=is_bf16)\n\n    for _ in range(cfg["warmup"]): sa3_core()\n    torch.cuda.synchronize()\n    stop,peak,thr=_smi_peak()\n    times=[]\n    for _ in range(cfg["iters"]):\n        t0=time.perf_counter(); sa3_core(); torch.cuda.synchronize()\n        times.append((time.perf_counter()-t0)*1000)\n    stop.set(); thr.join(timeout=2)\n    med=statistics.median(times)\n    flops=4*cfg["batch"]*cfg["heads"]*(padded_seq**2)*cfg["hd"]\n    tf=round(flops/(med/1000)/1e12,3)\n\nprint(json.dumps({"mean_ms":round(statistics.mean(times),4),\n    "median_ms":round(med,4),\n    "min_ms":round(min(times),4),"max_ms":round(max(times),4),\n    "stdev_ms":round(statistics.stdev(times),4),\n    "peak_vram_mib":peak[0],"tflops":tf,\n    "padded_seq":padded_seq}))\n'

def bench_sa3_direct(batch, heads, seq, hd, dtype, warmup, iters, q=None, k=None, v=None):
    """Run SA3 benchmark directly in same process (no subprocess).

    q, k, v may be passed in from the caller to avoid double-allocating VRAM.
    If not provided they are allocated locally (subprocess-like mode).
    """
    monitor = VramMonitor(interval=0.05)
    with torch.no_grad():
        # if no tensors passed in, allocate them locally — caller-owned
        # tensors are reused as-is and not freed by this function
        if q is None:
            q = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
            k = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
            v = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
        qc, kc, vc, ds = preprocess_qkv(q, k, v)
        ql = scale_and_quant_fp4(qc)
        kl = scale_and_quant_fp4_permute(kc)
        vl = scale_and_quant_fp4_transpose(vc)
        KL = qc.shape[2]
        padded_seq = qc.shape[2]
        is_bf16 = (dtype == torch.bfloat16)

        def sa3_core():
            return blockscaled_fp4_attn(ql, kl, vl, ds, KL, is_causal=False, is_bf16=is_bf16)

        for _ in range(warmup): sa3_core()
        torch.cuda.synchronize()
        monitor.start()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter(); sa3_core(); torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    peak_vram_mib = monitor.stop()
    med = statistics.median(times)
    flops = 4 * batch * heads * (padded_seq ** 2) * hd  # kernel operates on padded_seq
    tf = round(flops / (med / 1000) / 1e12, 3)
    return {
        "mean_ms":      round(statistics.mean(times), 4),
        "median_ms":    round(med, 4),
        "min_ms":       round(min(times), 4),
        "max_ms":       round(max(times), 4),
        "stdev_ms":     round(statistics.stdev(times), 4),
        "peak_vram_mib": peak_vram_mib,
        "tflops":       tf,
        "padded_seq":   padded_seq,
    }


def bench_sa3(batch, heads, seq, hd, dtype_name, warmup, iters, timeout=120):
    cfg = json.dumps({"batch":batch,"heads":heads,"seq":seq,"hd":hd,
                      "dtype":dtype_name,"warmup":warmup,"iters":iters})
    try:
        r = subprocess.run([sys.executable,"-c",SA3_WORKER,cfg],
                           capture_output=True,text=True,timeout=timeout)
        if r.returncode==0 and r.stdout.strip():
            return json.loads(r.stdout.strip().splitlines()[-1])
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown"
        return {"error": err}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

# ── env info ──────────────────────────────────────────────────────────────────
def env_info():
    props = torch.cuda.get_device_properties(0)
    info = {
        "python":        sys.version.split()[0],
        "torch":         torch.__version__,
        "cuda_rt":       torch.version.cuda or "n/a",
        "gpu":           torch.cuda.get_device_name(0),
        "vram_gb":       2 ** round(math.log2(props.total_memory / (1024**3))),
        "compute_cap":   f"sm_{props.major}{props.minor}",
        "platform":      platform.platform(),
        "sageattention": "n/a",
        "sageattn3":     "n/a",
        "triton":        "n/a",
    }
    try:
        import importlib.metadata as m
        info["sageattention"] = m.version("sageattention")
    except: pass
    try:
        import importlib.metadata as m
        info["sageattn3"] = m.version("sageattn3")
    except: pass
    try:
        import triton; info["triton"] = triton.__version__
    except: pass
    return info

# ── main loop ─────────────────────────────────────────────────────────────────
def run(warmup=10, iters=50):
    print("=" * 68)
    print("SageAttention Model Benchmark")
    print("=" * 68)
    env = env_info()
    for k_env, v_env in env.items():
        print(f"  {k_env:16s}: {v_env}")
    print()
    print("Kernel sources (verified real CUDA extensions):")
    print(f"  SA2        : {SA2_FILE}")
    print(f"  SA2-fp8    : {SA2_FP8_FILE}")
    print(f"  SA3        : {SA3_FILE}")
    print(f"  SDPA       : torch.nn.functional.scaled_dot_product_attention")
    print()
    print(f"  SA2        : {'OK' if HAS_SA2     else 'NOT FOUND'}")
    print(f"  SA2-fp8    : {'OK' if HAS_SA2_FP8 else 'NOT FOUND'}")
    print(f"  SA3-FP4    : {'OK (direct)' if HAS_SA3 and not SA3_SUBPROCESS else ('OK (subprocess)' if HAS_SA3 else 'NOT FOUND')}")
    print()

    total_vram_gb = 2 ** round(math.log2(torch.cuda.get_device_properties(0).total_memory / (1024**3)))
    results = {"env": env, "configs": []}

    # initialise q/k/v to None so cleanup with `del` is always safe
    # (avoids try/except NameError pattern)
    q = k = v = None

    for batch, heads, seq, hd, label, dtype in CONFIGS:
        dtype_name = "fp16" if dtype == torch.float16 else "bf16"
        print(f"── {label}")

        entry = {"label": label, "batch": batch, "heads": heads,
                 "seq": seq, "head_dim": hd, "dtype": dtype_name}

        # clear VRAM from previous config BEFORE checking availability
        if q is not None:
            del q, k, v
            q = k = v = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        needed    = vram_required_gb(batch, heads, seq, hd, dtype)
        available = vram_available_gb()
        hard_min  = get_min_vram_limit(label, HAS_SA3)

        if hard_min and total_vram_gb < hard_min:
            print(f"   [SKIP] requires {hard_min} GB VRAM, device has {total_vram_gb} GB\n")
            entry["skip"] = f"requires {hard_min}GB VRAM, device has {total_vram_gb}GB"
            results["configs"].append(entry)
            continue

        if needed > available:
            print(f"   [SKIP] not enough VRAM — need {needed:.2f} GB, available {available:.2f} GB\n")
            entry["skip"] = f"OOM: need {needed:.2f}GB, available {available:.2f}GB"
            results["configs"].append(entry)
            continue

        try:
            q = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
            k = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
            v = torch.randn(batch, heads, seq, hd, dtype=dtype, device="cuda")
        except Exception as e:
            print(f"   [OOM] {e}\n")
            q = k = v = None
            results["configs"].append(entry)
            continue

        # SA2
        if HAS_SA2:
            try:
                t, peak_vram_mib = bench(sageattn, q, k, v, tensor_layout="HND",
                          is_causal=False, warmup=warmup, iters=iters)
                entry["sa2"] = stats(t, peak_vram_mib, batch, heads, seq, hd)
                print(f"   sa2      : {entry['sa2']['median_ms']:8.3f} ms  "
                      f"min {entry['sa2']['min_ms']:.3f}  "
                      f"stdev {entry['sa2']['stdev_ms']:.3f}  "
                      f"vram {entry['sa2']['peak_vram_mib']}MiB  "
                      f"{entry['sa2']['tflops']:.3f} TFLOPS")
            except Exception as e:
                entry["sa2"] = {"error": str(e).splitlines()[0]}
                print(f"   sa2      : ERR {str(e).splitlines()[0]}")

        # SA2 fp8
        if HAS_SA2_FP8:
            try:
                t, peak_vram_mib = bench(sageattn_qk_int8_pv_fp8_cuda, q, k, v,
                          tensor_layout="HND", is_causal=False,
                          warmup=warmup, iters=iters)
                entry["sa2_fp8"] = stats(t, peak_vram_mib, batch, heads, seq, hd)
                print(f"   sa2_fp8  : {entry['sa2_fp8']['median_ms']:8.3f} ms  "
                      f"min {entry['sa2_fp8']['min_ms']:.3f}  "
                      f"stdev {entry['sa2_fp8']['stdev_ms']:.3f}  "
                      f"vram {entry['sa2_fp8']['peak_vram_mib']}MiB  "
                      f"{entry['sa2_fp8']['tflops']:.3f} TFLOPS")
            except Exception as e:
                entry["sa2_fp8"] = {"error": str(e).splitlines()[0]}
                print(f"   sa2_fp8  : ERR {str(e).splitlines()[0]}")

        # SDPA — run BEFORE SA3 so we can free q/k/v before subprocess SA3
        # allocates its own tensors (avoids double-allocation OOM on edge cases
        # like HunyuanVideo 1920x1088 where Q+K+V alone is ~33 GB)
        try:
            t, peak_vram_mib = bench(F.scaled_dot_product_attention, q, k, v,
                      is_causal=False, warmup=warmup, iters=iters)
            entry["sdpa"] = stats(t, peak_vram_mib, batch, heads, seq, hd)
            print(f"   sdpa     : {entry['sdpa']['median_ms']:8.3f} ms  "
                  f"min {entry['sdpa']['min_ms']:.3f}  "
                  f"stdev {entry['sdpa']['stdev_ms']:.3f}  "
                  f"vram {entry['sdpa']['peak_vram_mib']}MiB  "
                  f"{entry['sdpa']['tflops']:.3f} TFLOPS")
        except Exception as e:
            entry["sdpa"] = {"error": str(e).splitlines()[0]}
            print(f"   sdpa     : ERR {str(e).splitlines()[0]}")

        # SA3 FP4 (direct or subprocess, core kernel only)
        if HAS_SA3:
            if hd != 128:
                entry["sa3"] = {"error": f"SA3 requires hd=128, got hd={hd}"}
                print(f"   sa3      : SKIP (hd={hd}, SA3 requires hd=128)")
            else:
                print(f"   sa3      : running...", end="", flush=True)
                if SA3_SUBPROCESS:
                    # free main-process tensors so subprocess has full VRAM available
                    del q, k, v
                    q = k = v = None
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    r = bench_sa3(batch, heads, seq, hd, dtype_name, warmup, iters)
                else:
                    # direct mode reuses already-allocated q/k/v — no double allocation
                    r = bench_sa3_direct(batch, heads, seq, hd, dtype, warmup, iters, q=q, k=k, v=v)
                entry["sa3"] = r
                if "error" in r:
                    print(f" ERR {r['error'].splitlines()[0]}")
                else:
                    pad_note = ""
                    if r.get("padded_seq") and r["padded_seq"] != seq:
                        pad_note = f"  (kernel seq={r['padded_seq']}, padded from {seq})"
                    vram_note = f"  vram {r.get('peak_vram_mib','')}MiB" if "peak_vram_mib" in r else ""
                    tflops_note = f"  {r['tflops']:.3f} TFLOPS" if "tflops" in r else ""
                    print(f" {r['median_ms']:8.3f} ms  "
                          f"min {r['min_ms']:.3f}  stdev {r['stdev_ms']:.3f}"
                          f"{vram_note}{tflops_note}{pad_note}")

        results["configs"].append(entry)
        print()

    # final cleanup
    if q is not None:
        del q, k, v
        q = k = v = None
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters",  type=int, default=50)
    args = p.parse_args()

    data = run(warmup=args.warmup, iters=args.iters)

    out = Path(gpu_filename())
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n[DONE] Saved: {out.resolve()}")