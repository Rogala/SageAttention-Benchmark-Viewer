# SageAttention Benchmark Viewer

A three-part toolkit for measuring and comparing attention kernel performance across the exact attention configurations used by generative models running inside ComfyUI.

**Live viewer:** https://rogala.github.io/SageAttention-Benchmark-Viewer/

---

## What this is — and what it is not

This is not a benchmark of image or video generation speed. No model weights are loaded, no inference pipeline runs.

The benchmark operates on randomly generated tensors that match the exact shapes — `batch × heads × seq_len × head_dim`, dtype — that real models produce during sampling. This distinction matters: the numbers reflect raw attention kernel throughput for those specific shapes, not end-to-end generation time.

Attention in transformer-based generative models is computed as:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

The computational cost scales quadratically with sequence length (`seq_len²`), which makes the attention kernel the dominant bottleneck at high resolutions and long video durations. The standard formula for attention FLOPS is:

```
FLOPS = 4 × B × H × S² × D
```

where `B` = batch, `H` = heads, `S` = sequence length, `D` = head dimension. This is the formula used throughout the codebase to compute TFLOPS figures. The derivation is discussed in the FlashAttention-2 paper ([Dao, 2023](https://arxiv.org/abs/2307.08691)).

Four kernel variants are compared against each model's attention shapes:

- **SA2** — SageAttention 2, INT8 quantized QK with FP16/BF16 PV ([paper](https://arxiv.org/abs/2411.10958))
- **SA2-fp8** — SageAttention 2 with FP8 PV accumulation
- **SA3-FP4** — SageAttention 3, block-scaled FP4 attention ([paper](https://arxiv.org/abs/2505.11594))
- **SDPA** — PyTorch `scaled_dot_product_attention` (FlashAttention-2 backend), used as baseline

---

<img width="1920" height="871" alt="Знімок екрана (55)" src="https://github.com/user-attachments/assets/ae56492f-966b-4747-b51e-da2d51ef5584" />

---
## Covered models

**Image:** SDXL-1.0, SD3.5-Large, Flux.1-Dev (Kontext / Krea), Flux.2-Dev, Flux.2-Dev Klein 9B, Z-Image / Z-Image Turbo, Qwen-Image-2512, Qwen-Image-Edit-2511, ERNIE-Image / ERNIE-Image Turbo

**Video:** LTX-2.3, Wan2.2, HunyuanVideo-1.5

**Audio:** ACE-Step-1.5

---

## How attention shapes were collected — `attention_logger_node.py`

A ComfyUI custom node that hooks into `comfy.ldm.modules.attention.optimized_attention` and logs every unique combination of `(heads, head_dim, seq_len, dtype)` that passes through during a real sampling run.

Two interception modes:

- **override mode** (default) — installs itself via `optimized_attention_override` inside the model's `transformer_options`. Used for SDXL, SD3.5, Flux, Wan2.2, HunyuanVideo, LTX, and others.
- **patch_global mode** — patches `comfy_attn.optimized_attention` at the module level, then re-patches any submodule that imported the function directly. Required for models that bypass the override mechanism: ERNIE-Image and ACE-Step.

**Workflow A (standard models):**
```
Load Model → [Attention Logger rogala] → KSampler
```

**Workflow B (ERNIE / ACE-Step):**
```
Load Model → [Attention Logger rogala | patch_global=True] → KSampler
```

Each unique shape is printed once to the ComfyUI console. The collected output for all covered models is in `input_data.txt`.

---

## How the data was processed — `input_data.txt`

The raw logger output was manually organized by model and resolution. Each line has the form:

```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  4352  dtype=torch.bfloat16
```

From these values, benchmark configurations were constructed as tuples:

```python
(batch, heads, seq, head_dim, label, dtype)
```

`batch` is always 1 — ComfyUI runs single-sample inference. For models with multiple attention layers per step (Qwen-Image-Edit, LTX cross-attention), only self-attention shapes with `hd=128` are benchmarked; cross-attention layers with `hd=64` are excluded because SA3 requires `hd=128` and the comparison would be inconsistent. For HunyuanVideo and Wan2.2, which produce two seq values per resolution (double-pass architecture), the larger value is taken as the worst-case shape.

---

## How the benchmark runs

```bash
python bench_windows.py
python bench_windows.py --warmup 20 --iters 100

python3 bench_linux.py
python3 bench_linux.py --warmup 20 --iters 100
```

For each configuration the script:

1. Checks VRAM availability. The minimum memory footprint for Q+K+V tensors is estimated as `batch × heads × seq × head_dim × 3 × 2 bytes`. Configs that exceed free VRAM are skipped and recorded as `OOM` in the output. Configs requiring SA3 FP4 quantization buffers have additional hard minimums defined in `VRAM_LIMITS`.
2. Allocates random `(B, H, S, D)` tensors in the target dtype on CUDA.
3. Runs `warmup` iterations (default 10) to populate CUDA caches and stabilize clock frequencies.
4. Runs `iters` timed iterations (default 50) with `torch.cuda.synchronize()` after each call.
5. Records median, mean, min, max, stdev in milliseconds; peak VRAM during timed iterations; and TFLOPS computed from the median time.

SA3-FP4 preprocessing (`preprocess_qkv`, quantization) is excluded from the timed region — only the core `blockscaled_fp4_attn` kernel call is measured. SA3 pads `seq_len` to the nearest multiple of 128 internally; the padded value is reported separately so TFLOPS are computed on the actual kernel dimensions.

When SA2 and SA3 conflict in the same Python process, SA3 runs via subprocess to isolate the environment.

**Linux note:** the Linux build uses `pynvml` (`nvidia-ml-py`) for VRAM monitoring instead of `nvidia-smi` subprocess polling. On Linux, each `subprocess.run()` call triggers a `fork()` + `exec()` which has measurable overhead at 50 ms polling intervals. pynvml queries the driver directly via shared library call. Falls back to `nvidia-smi` if pynvml is not installed.

Output is saved automatically to a JSON file named after the GPU and VRAM:

```
5060-ti-16.json
4070-ti_super-16.json
3060-laptop-6.json
```

---

## How to view results — `viewer.html`

Open `viewer.html` in any browser — no server required. Alternatively use the live version at https://rogala.github.io/SageAttention-Benchmark-Viewer/

- Load one or more JSON result files. Multiple GPUs can be compared side by side in the same chart.
- Filter by model category (Image / Video / Audio).
- Toggle between latency (ms) and throughput (TFLOPS) views.
- Toggle individual kernels on or off.
- The community database dropdown loads results submitted by other users directly from this repository.

---

## How to share results

Run `bench_windows.py` or `bench_linux.py` on your GPU. The output JSON file is named automatically. Submit it as a pull request to the `/results` directory, or attach it to an issue. Results from different GPUs can be loaded simultaneously in the viewer for direct comparison.

---

## Requirements

- CUDA GPU (NVIDIA)
- Python 3.10+
- PyTorch with CUDA
- `sageattention` — SA2 / SA2-fp8
- `sageattn3` — SA3-FP4 (optional)
- `nvidia-smi` accessible in PATH
- `nvidia-ml-py` — recommended on Linux (`pip install nvidia-ml-py`)

---

## References

- Tri Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023 — https://arxiv.org/abs/2307.08691
- Jintao Zhang et al., "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration," 2024 — https://arxiv.org/abs/2411.10958
- Jintao Zhang et al., "SageAttention3," 2025 — https://arxiv.org/abs/2505.11594

---

## Acknowledgements

[Jukka Seppänen / kijai](https://github.com/kijai/ComfyUI-KJNodes) — for the PatchSageAttentionKJ node which inspired the override pattern used in `attention_logger_node.py`.

[woct0rdho](https://github.com/woct0rdho) — for the Windows forks [triton-windows](https://github.com/triton-lang/triton-windows) and [SageAttention](https://github.com/woct0rdho/SageAttention) (SA2 / SA3).

[mengqin](https://github.com/mengqin/SageAttention) — for the [SageAttention](https://github.com/mengqin/SageAttention) Windows fork with SA3 support and build fixes.

Built with the assistance of [Claude](https://claude.ai).
