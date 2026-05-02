# Attention Shapes Reference

This file contains attention tensor shapes logged from real ComfyUI sampling runs across all benchmarked models. These are the exact shapes used by `bench_windows.py` and `bench_linux.py`.

## How this data was collected

Each line was produced by `attention_logger_node.py` — a ComfyUI custom node that hooks into `optimized_attention` and prints every unique tensor shape that passes through during sampling.

## How to read a line

```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  4352  dtype=torch.bfloat16
```

| Field   | Meaning                                          |
|---------|--------------------------------------------------|
| `heads` | Number of attention heads                        |
| `hd`    | Head dimension (channels per head)               |
| `seq`   | Sequence length (tokens)                         |
| `dtype` | Tensor dtype (`torch.float16` / `torch.bfloat16`) |

Lines tagged `| global` come from global-patch mode (ERNIE-Image and ACE-Step).

## Notes on shape selection

Not every logged shape ends up in the benchmark. Excluded:

- **Cross-attention with `hd=64` (LTX)** — SA3 requires `hd=128`, cross-attn excluded for consistency
- **Text-encoder embeddings** (`seq=32` for Z-Image, `seq=77` for HunyuanVideo) — tiny shapes, not dominant load
- **Second-pass duplicates** (Wan2.2, HunyuanVideo) — same shape both passes; benchmark uses worst-case (largest)

---

## Image Models

### SDXL-1.0

Resolutions: `1024×1024`, `1344×768`

```
[ATTN LOGGER rogala] heads= 10  hd=  64  seq=  4096  dtype=torch.float16
[ATTN LOGGER rogala] heads= 20  hd=  64  seq=  1024  dtype=torch.float16
[ATTN LOGGER rogala] heads= 10  hd=  64  seq=  4032  dtype=torch.float16
[ATTN LOGGER rogala] heads= 20  hd=  64  seq=  1008  dtype=torch.float16
```

---

### SD3.5-Large

Resolutions: `1024×1024`, `1344×768`, `1152×1152`, `1536×832`

```
[ATTN LOGGER rogala] heads= 38  hd=  64  seq=  4250  dtype=torch.float16
[ATTN LOGGER rogala] heads= 38  hd=  64  seq=  4186  dtype=torch.float16
[ATTN LOGGER rogala] heads= 38  hd=  64  seq=  5338  dtype=torch.float16
[ATTN LOGGER rogala] heads= 38  hd=  64  seq=  5146  dtype=torch.float16
```

---

### Flux.1-Dev (Kontext, Krea)

Resolutions: `1024×1024`, `1344×768`, `1216×1216`, `1664×896`

```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  4352  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  4288  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6032  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6080  dtype=torch.bfloat16
```

---

### Flux.2-Dev

Resolutions: `1024×1024`, `1344×768`, `1440×1440`, `1920×1088`

```
[ATTN LOGGER rogala] heads= 48  hd= 128  seq=  4608  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 48  hd= 128  seq=  4544  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 48  hd= 128  seq=  8612  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 48  hd= 128  seq=  8672  dtype=torch.bfloat16
```

---

### Flux.2-Dev (Klein 9B, Klein 9B-kv)

Resolutions: `1024×1024`, `1344×768`, `1440×1440`, `1920×1088`

```
[ATTN LOGGER rogala] heads= 32  hd= 128  seq=  4356  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd= 128  seq=  4413  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd= 128  seq=  8793  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd= 128  seq=  8740  dtype=torch.bfloat16
```

---

### Z-Image (Turbo)

Resolutions: `1024×1024`, `1344×768`, `1280×1280`, `1600×896`

Joint attention: img tokens + 32 txt tokens concatenated. Only the joint shape (last in each block) is benchmarked.

**1024×1024:**
```
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=    32  dtype=torch.bfloat16   <- text only
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  4096  dtype=torch.bfloat16   <- img only
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  4128  dtype=torch.bfloat16   <- joint (benchmarked)
```

**1344×768:**
```
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=    32  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  4032  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  4064  dtype=torch.bfloat16
```

**1280×1280:**
```
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=    32  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  6400  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  6432  dtype=torch.bfloat16
```

**1600×896:**
```
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=    32  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  5600  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 30  hd= 128  seq=  5632  dtype=torch.bfloat16
```

---

### Qwen-Image-2512

Resolutions: `1328×1328`, `1664×928`, `1920×1920`, `2560×1440`

```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6978  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6121  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14489  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14489  dtype=torch.bfloat16
```

---

### Qwen-Image-Edit-2511

Resolutions: `1328×1328`, `1664×928`, `1920×1920`, `2560×1440`

Two attention layers per step — the larger `seq` value is taken per resolution.

**1328×1328:**
```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6912  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6894  dtype=torch.bfloat16
```

**1664×928:**
```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6055  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq=  6037  dtype=torch.bfloat16
```

**1920×1920:**
```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14423  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14405  dtype=torch.bfloat16
```

**2560×1440:**
```
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14423  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 24  hd= 128  seq= 14405  dtype=torch.bfloat16
```

---

### ERNIE-Image (Turbo)

Resolutions: `1024×1024`, `1344×768`

Captured via `patch_global=True` — ERNIE bypasses the standard override mechanism.

```
[ATTN LOGGER rogala | global] heads= 32  hd= 128  seq=  4097  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 32  hd= 128  seq=  4033  dtype=torch.bfloat16
```

---

## Video Models

### LTX-2.3

Resolutions: `960×544`, `1280×736`, `1600×960`, `1920×1088`

Only `hd=128` main self-attention is benchmarked. `hd=64` cross-attention excluded (SA3 requires `hd=128`).

**960×544:**
```
[ATTN LOGGER rogala] heads= 32  hd= 128  seq=  8160  dtype=torch.bfloat16   <- benchmarked
[ATTN LOGGER rogala] heads= 32  hd=  64  seq=   127  dtype=torch.bfloat16   <- cross-attn (skipped)
[ATTN LOGGER rogala] heads= 32  hd=  64  seq=  8160  dtype=torch.bfloat16   <- cross-attn (skipped)
```

**1280×736:**
```
[ATTN LOGGER rogala] heads= 32  hd= 128  seq= 14720  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq=   127  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq= 14720  dtype=torch.bfloat16
```

**1600×960:**
```
[ATTN LOGGER rogala] heads= 32  hd= 128  seq= 24000  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq=   127  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq= 24000  dtype=torch.bfloat16
```

**1920×1088:**
```
[ATTN LOGGER rogala] heads= 32  hd= 128  seq= 32640  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq=   127  dtype=torch.bfloat16
[ATTN LOGGER rogala] heads= 32  hd=  64  seq= 32640  dtype=torch.bfloat16
```

---

### Wan2.2

Resolutions: `832×480`, `960×544`, `1280×720`, `1920×1088`

Two passes per step, sequential — same `seq` for both passes. Benchmark times one pass (worst case = second pass).

**832×480:**
```
[ATTN LOGGER rogala] heads= 40  hd= 128  seq=  7800  dtype=torch.float16
[ATTN LOGGER rogala] heads= 40  hd= 128  seq=  7800  dtype=torch.float16
```

**960×544:**
```
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 10200  dtype=torch.float16
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 10200  dtype=torch.float16
```

**1280×720:**
```
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 18000  dtype=torch.float16
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 18000  dtype=torch.float16
```

**1920×1088:**
```
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 40800  dtype=torch.float16
[ATTN LOGGER rogala] heads= 40  hd= 128  seq= 40800  dtype=torch.float16
```

---

### HunyuanVideo-1.5

Resolutions: `848×480`, `1280×720`, `1920×1088`

Two close `seq` values per resolution — larger is taken as worst-case.  
Auxiliary shapes (`seq=77` text emb, `seq=6` temporal) are not benchmarked.

**848×480:**
```
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=    77  dtype=torch.float16   <- text emb (skipped)
[ATTN LOGGER rogala] heads= 16  hd= 128  seq= 49367  dtype=torch.float16   <- benchmarked
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=     6  dtype=torch.float16   <- temporal (skipped)
[ATTN LOGGER rogala] heads= 16  hd= 128  seq= 49296  dtype=torch.float16   <- second pass
```

**1280×720:**
```
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=    77  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=111677  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=     6  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=111606  dtype=torch.float16
```

**1920×1088:**
```
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=    77  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=253037  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=     6  dtype=torch.float16
[ATTN LOGGER rogala] heads= 16  hd= 128  seq=252966  dtype=torch.float16
```

---

## Audio Models

### ACE-Step-1.5

Durations: `50s`, `100s`, `150s`

Captured via `patch_global=True`. `seq=326` is fixed text/lyrics encoder. Audio `seq` scales with duration. `seq=5` auxiliary shape is skipped.

**50s:**
```
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=   326  dtype=torch.bfloat16   <- text-attn
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=  1250  dtype=torch.bfloat16   <- audio
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=     5  dtype=torch.bfloat16   <- skipped
[ATTN LOGGER rogala | global] heads= 32  hd= 128  seq=   625  dtype=torch.bfloat16   <- cross
```

**100s:**
```
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=   326  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=  2500  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=     5  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 32  hd= 128  seq=  1250  dtype=torch.bfloat16
```

**150s:**
```
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=   326  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=  3750  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 16  hd= 128  seq=     5  dtype=torch.bfloat16
[ATTN LOGGER rogala | global] heads= 32  hd= 128  seq=  1875  dtype=torch.bfloat16
```
