"""
attention_logger_node.py
ComfyUI/custom_nodes/attention_logger_node.py

Logs unique attention shapes (heads, head_dim, seq_len, dtype) that pass through
ComfyUI's optimized_attention during sampling. Useful for profiling and benchmarking
real attention configurations of generative models.

Two modes:
- override:      hooks into optimized_attention_override (compatible with SageAttention
                 nodes) — use for standard models (SD, SDXL, Flux, SD3, etc.)
- patch_global:  patches optimized_attention globally at module level — use for models
                 that bypass the override mechanism (ERNIE, ACE-Step, etc.)

Workflow A (standard models):
  Load Model -> [Attention Logger] -> KSampler

Workflow B (ERNIE / ACE-Step):
  Load Model -> [Attention Logger patch_global=True] -> KSampler

Inspired by Kijai's KJNodes attention override pattern.
"""

import time
import sys
import torch
import comfy.ldm.modules.attention as comfy_attn

# Keep a reference to the original function so we can restore it after global patching.
# Must be captured at import time — before any patch is applied.
_original_optimized = comfy_attn.optimized_attention
_global_patch_active = False


def _restore_global_patch():
    """Restore the original optimized_attention before applying a new patch.
    Called at the start of every apply() to avoid stacking patches across runs."""
    global _global_patch_active
    comfy_attn.optimized_attention = _original_optimized
    _global_patch_active = False


def _apply_global_patch(seen):
    """Replace comfy_attn.optimized_attention with a logging wrapper globally.

    Also back-patches any submodules (ERNIE, ACE-Step) that imported
    optimized_attention directly into their own namespace at load time —
    those modules hold a local reference and won't see the global reassignment
    unless we patch them individually.
    """
    global _global_patch_active
    original = _original_optimized

    def patched(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False,
                low_precision_attention=True, **kwargs):

        # When skip_reshape=True the tensor is already split into heads,
        # so shape is (batch, heads, seq, dim_head) — read dim and seq directly.
        # When skip_reshape=False the heads are still packed:
        # shape is (batch, seq, heads * dim_head) — divide to get dim_head.
        if skip_reshape:
            dim_head = q.shape[-1]
            seq = q.shape[-2]
        else:
            dim_head = q.shape[-1] // heads
            seq = q.shape[1]

        # Use a tuple as a hashable key so each unique shape is logged only once.
        key = (heads, dim_head, seq, str(q.dtype))
        if key not in seen:
            seen.add(key)
            print(f"[ATTN LOGGER rogala | global] heads={heads:3d}  "
                  f"hd={dim_head:4d}  "
                  f"seq={seq:6d}  "
                  f"dtype={q.dtype}")

        return original(q, k, v, heads, mask=mask,
                        attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                        low_precision_attention=low_precision_attention,
                        **kwargs)

    comfy_attn.optimized_attention = patched

    # ERNIE imports optimized_attention at module load time into its own namespace,
    # so patching comfy_attn alone is not enough — patch the reference inside ernie too.
    try:
        import comfy.ldm.ernie.model as ernie_model
        ernie_model.optimized_attention = patched
        print("[ATTN LOGGER rogala] patched ernie model")
    except Exception:
        pass

    # ACE-Step may be spread across several submodules — scan sys.modules and
    # patch any that hold their own reference to optimized_attention.
    try:
        for mod_name, mod in list(sys.modules.items()):
            if "ace_step" in mod_name.lower() and hasattr(mod, "optimized_attention"):
                mod.optimized_attention = patched
                print(f"[ATTN LOGGER rogala] patched {mod_name}")
    except Exception:
        pass

    _global_patch_active = True


class AttentionLoggerRogala:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "patch_global": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Patch optimized_attention globally. "
                        "Use for ERNIE, ACE-Step and other models that bypass the override mechanism."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "rogala/debug"
    DESCRIPTION = (
        "Logs unique attention tensor shapes (heads, head_dim, seq_len, dtype) "
        "to the console during sampling. "
        "Enable patch_global for ERNIE / ACE-Step models."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always return a changing value so ComfyUI re-executes this node on
        # every queue — necessary to reset `seen` and re-log shapes after
        # switching models or resolutions.
        return time.time()

    def apply(self, model, enabled, patch_global):
        global _global_patch_active

        # Always clean up a previously installed global patch first.
        # Without this, re-running with different settings stacks wrappers.
        if _global_patch_active:
            _restore_global_patch()

        if not enabled:
            print("[ATTN LOGGER rogala] disabled")
            return (model,)

        # Fresh set per run — ensures shapes are re-logged after model/resolution changes.
        seen = set()

        if patch_global:
            _apply_global_patch(seen)
            print("[ATTN LOGGER rogala] global patch applied")
            return (model,)

        # Override mode: inject via transformer_options so the logger sits in the
        # same override chain as SageAttention KJ nodes without breaking them.
        model_clone = model.clone()

        # Capture any upstream override (e.g. SageAttention) so we can forward
        # to it after logging — preserves the full override chain.
        existing_override = model_clone.model_options.get(
            "transformer_options", {}
        ).get("optimized_attention_override", None)

        def attention_override_with_log(func, q, k, v, heads, mask=None,
                                        attn_precision=None, skip_reshape=False,
                                        skip_output_reshape=False, **kwargs):
            # Same shape extraction logic as the global patch (see comments above).
            if skip_reshape:
                dim_head = q.shape[-1]
                seq = q.shape[-2]
            else:
                dim_head = q.shape[-1] // heads
                seq = q.shape[1]

            key = (heads, dim_head, seq, str(q.dtype))
            if key not in seen:
                seen.add(key)
                print(f"[ATTN LOGGER rogala] heads={heads:3d}  "
                      f"hd={dim_head:4d}  "
                      f"seq={seq:6d}  "
                      f"dtype={q.dtype}")

            if existing_override is not None:
                # Forward to SageAttention (or any other upstream override).
                return existing_override(func, q, k, v, heads, mask=mask,
                                         attn_precision=attn_precision,
                                         skip_reshape=skip_reshape,
                                         skip_output_reshape=skip_output_reshape,
                                         **kwargs)
            else:
                # No upstream override — call the base attention function directly.
                return func(q, k, v, heads, mask=mask,
                            attn_precision=attn_precision,
                            skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape,
                            **kwargs)

        model_clone.model_options.setdefault("transformer_options", {})
        model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_with_log

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "AttentionLoggerRogala": AttentionLoggerRogala,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttentionLoggerRogala": "Attention Logger (rogala)",
}
