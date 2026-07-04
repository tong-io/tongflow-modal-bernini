# tongflow-modal-bernini

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Unified video / image
generation and editing with **Bernini-R 1.3B**
([`ByteDance/Bernini-R-1.3B-Diffusers`](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers),
Apache-2.0) — a single Wan2.1-1.3B-based diffusion renderer — running on a single
A100-80GB GPU via [Modal](https://modal.com).

Upstream: [bytedance/Bernini](https://github.com/bytedance/Bernini).

## Capabilities

One model serves the whole task family. This plugin maps it to these ABI slots:

| Slot | Task | What it does |
| --- | --- | --- |
| `image-gen` | t2i | Text → image |
| `image-edit` | i2i | Image + instruction → image |
| `text-gen-video` | t2v | Text → video |
| `image-gen-video` | r2v | Reference image → video |
| `video-image-gen-video-mix` | rv2v | Reference image + video + instruction → video |
| `video-edit` | v2v | Video + instruction → video (general editing) |
| `subtitle_remove` | v2v | Remove subtitles / captions from a video |
| `remove_watermark` | v2v | Remove watermarks / logos from a video |

`subtitle_remove` and `remove_watermark` are video editing under the hood, driven by a
fixed internal instruction — they are among Bernini-R's strongest simple-task cases.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | Optional | Only helps avoid Hugging Face rate limits; Bernini-R weights are public. |

On first use the plugin deploys to your Modal account automatically and caches the build.
If set in Settings, `HF_TOKEN` is injected into the Modal download job at deploy time — no
manual `modal secret create` needed.

## Notes

- **Single GPU, single A100-80GB** — the 1.3B renderer fits comfortably; no torchrun /
  Ulysses sequence parallelism is used (mirrors upstream `infer_single_gpu.py`).
- **FlashAttention is omitted** by default; Bernini falls back to PyTorch SDPA. Add
  `flash-attn==2.8.3` to the image in `deploy.py` for a speedup once the base deploy is green.
- The per-task `guidance_mode` / `num_frames` in `deploy.py` are taken from the upstream
  `scripts/bernini_r/run_*.sh`; tune there if a future release changes the recipe.
