"""Modal deploy entry for the Bernini-R (1.3B) unified video renderer.

Wraps ByteDance's Bernini Renderer (https://github.com/bytedance/Bernini,
Apache-2.0) — a single Wan2.1-1.3B-based diffusion renderer that covers the
t2i / i2i / t2v / v2v / rv2v / r2v task family — as TongFlow ABI node slots.

Deploy:
    modal deploy deploy.py

Notes:
  - Single A100-80GB, single GPU: the 1.3B renderer fits comfortably; no
    torchrun / Ulysses sequence parallelism is needed (mirrors the repo's
    ``infer_single_gpu.py``, which does no ``init_process_group``).
  - The per-task ``guidance_mode`` / ``num_frames`` below are taken verbatim
    from the repo's ``scripts/bernini_r/run_*.sh``; tune them here if a future
    Bernini release changes the recommended sampling recipe.
  - FlashAttention is intentionally omitted: Bernini auto-falls back to PyTorch
    SDPA when ``flash_attn`` is absent, which keeps the image build simple.
    Add ``flash-attn==2.8.3`` to the image for a speedup once the base deploy
    is green.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.remove_watermark import (
    RemoveWatermarkInput,
    RemoveWatermarkOutput,
)
from tongflow.models.subtitle_remove import (
    SubtitleRemoveInput,
    SubtitleRemoveOutput,
)
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.video_edit import VideoEditInput, VideoEditOutput
from tongflow.models.video_image_gen_video_mix import (
    VideoImageGenVideoMixInput,
    VideoImageGenVideoMixOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import node_slot

# ── model / weights ───────────────────────────────────────────────────────────

# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["video-edit"]

REPO_ID = "ByteDance/Bernini-R-1.3B-Diffusers"
MODEL_DIR = f"/models/{REPO_ID}"

# Sampling recipe — plugin-internal, NOT part of the ABI contract. Values match
# the renderer defaults in the upstream CLI (bernini/cli.py).
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
NUM_INFERENCE_STEPS = 40
OMEGA_VID = 1.25
OMEGA_IMG = 4.5
OMEGA_TXT = 4.0
OMEGA_SCALE = 0.8
FLOW_SHIFT = 5.0
FPS = 16
VIDEO_FRAMES = 81
MAX_TRAINED_SRC_ID = 5

# Fixed instructions for the two dedicated cleanup slots (v2v under the hood).
SUBTITLE_PROMPT = (
    "Remove all subtitles and on-screen captions from the video. "
    "Keep every other element, motion, and background unchanged."
)
WATERMARK_PROMPT = (
    "Remove all watermarks, logos, and station bugs from the video. "
    "Keep every other element, motion, and background unchanged."
)

volume = modal.Volume.from_name("models", create_if_missing=True)


# ── app / image ───────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "tongflow==0.2.13", "fastapi[standard]",
        "diffusers==0.35.2",
        "accelerate==0.34.2",
        "transformers==4.57.3",
        "safetensors",
        "einops",
        "numpy",
        "Pillow",
        "tqdm",
        "ftfy",
        "ninja",
        "scipy",
        "decord",
        "imageio",
        "imageio-ffmpeg",
        "huggingface_hub",
    )
    # Open-VeOmni and Bernini are git-only and must NOT drag in a different
    # torch build, so install both with --no-deps (their runtime deps are
    # already pinned above).
    .run_commands(
        "pip install --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10",
        "pip install --no-deps git+https://github.com/bytedance/Bernini.git@82d573e9",
    )
)

with image.imports():
    import torch
    from bernini import BerniniRendererPipeline


@deploy
@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=5,
)
class Inference:
    @modal.enter()
    def load(self) -> None:
        torch.cuda.set_device(torch.device("cuda:0"))
        # Diffusers layout: pass the directory as config and load the
        # transformer / transformer_2 weights from it (no high/low ckpt paths).
        self.pipe = BerniniRendererPipeline.from_pretrained(
            MODEL_DIR,
            device=torch.device("cuda:0"),
            load_ckpt_weights=False,
            use_unipc=True,
            shift=FLOW_SHIFT,
            use_src_id_rotary_emb=True,
            interpolate_src_id=True,
            max_trained_src_id=MAX_TRAINED_SRC_ID,
        )

    # ── shared runner ─────────────────────────────────────────────────────────

    def _run(
        self,
        prompt: str,
        *,
        guidance_mode: str,
        num_frames: int,
        out_suffix: str,
        video=None,
        image=None,
        images=None,
        seed: int = 42,
        height: int | None = None,
        width: int | None = None,
    ) -> bytes:
        """Materialise inputs to temp files, run one generation, return bytes.

        ``out_suffix`` is ``.png`` for single-frame (image) tasks and ``.mp4``
        for video tasks — Bernini's ``save_output`` switches on frame count.
        """
        import os
        import tempfile

        with ExitStack() as stack:
            video_path = (
                str(stack.enter_context(asset_as_path(video, suffix=".mp4")))
                if video is not None
                else None
            )
            image_path = (
                str(stack.enter_context(asset_as_path(image, suffix=".png")))
                if image is not None
                else None
            )
            image_paths = None
            if images:
                image_paths = [
                    str(stack.enter_context(asset_as_path(a, suffix=".png")))
                    for a in images
                ]

            out_fd, out_path = tempfile.mkstemp(suffix=out_suffix)
            os.close(out_fd)
            try:
                kwargs: dict = dict(
                    neg_prompt=NEG_PROMPT,
                    num_frames=num_frames,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_mode=guidance_mode,
                    omega_vid=OMEGA_VID,
                    omega_img=OMEGA_IMG,
                    omega_txt=OMEGA_TXT,
                    omega_scale=OMEGA_SCALE,
                    flow_shift=FLOW_SHIFT,
                    seed=seed,
                    fps=FPS,
                    video=video_path,
                    image=image_path,
                    images=image_paths,
                    output_path=out_path,
                    write_output=True,
                )
                if height is not None:
                    kwargs["height"] = height
                if width is not None:
                    kwargs["width"] = width
                self.pipe(prompt, **kwargs)
                return Path(out_path).read_bytes()
            finally:
                try:
                    os.unlink(out_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _seed(value) -> int:
        return int(value) if value is not None else 42

    # ── image slots ───────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        """t2i — text → image."""
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")
        raw = self._run(
            text,
            guidance_mode="t2v_apg",
            num_frames=1,
            out_suffix=".png",
            seed=self._seed(input.seed),
            height=input.height,
            width=input.width,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        """i2i — image + instruction → image."""
        text = (input.text or "").strip()
        if not text:
            return ImageEditOutput(success=False, error="Missing edit instruction")
        raw = self._run(
            text,
            guidance_mode="v2v",
            num_frames=1,
            out_suffix=".png",
            image=input.image,
            seed=self._seed(input.seed),
            height=input.height,
            width=input.width,
        )
        return ImageEditOutput(success=True, image=asset(raw, mime="image/png"))

    # ── video slots ───────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_VIDEO)
    def text_gen_video(self, input: TextGenVideoInput) -> TextGenVideoOutput:
        """t2v — text → video."""
        text = (input.text or "").strip()
        if not text:
            return TextGenVideoOutput(success=False, error="Missing text prompt")
        raw = self._run(
            text,
            guidance_mode="t2v_apg",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            seed=self._seed(input.seed),
            height=input.height,
            width=input.width,
        )
        return TextGenVideoOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_VIDEO)
    def image_gen_video(self, input: ImageGenVideoInput) -> ImageGenVideoOutput:
        """r2v — reference image → video."""
        raw = self._run(
            (input.text or "").strip(),
            guidance_mode="r2v_apg",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            images=[input.image],
            seed=self._seed(input.seed),
            height=input.height,
            width=input.width,
        )
        return ImageGenVideoOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MIX)
    def video_image_gen_video_mix(
        self, input: VideoImageGenVideoMixInput
    ) -> VideoImageGenVideoMixOutput:
        """rv2v — reference image + video + instruction → video."""
        raw = self._run(
            (input.text or "").strip(),
            guidance_mode="rv2v",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            video=input.video,
            images=[input.image],
        )
        return VideoImageGenVideoMixOutput(
            success=True, video=asset(raw, mime="video/mp4")
        )

    @modal.method()
    @node_slot(NodeSlots.VIDEO_EDIT)
    def video_edit(self, input: VideoEditInput) -> VideoEditOutput:
        """v2v — video + instruction → video."""
        text = (input.text or "").strip()
        if not text:
            return VideoEditOutput(success=False, error="Missing edit instruction")
        raw = self._run(
            text,
            guidance_mode="v2v_apg",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            video=input.video,
            seed=self._seed(input.seed),
            height=input.height,
            width=input.width,
        )
        return VideoEditOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.method()
    @node_slot(NodeSlots.SUBTITLE_REMOVE)
    def subtitle_remove(self, input: SubtitleRemoveInput) -> SubtitleRemoveOutput:
        """Subtitle removal — v2v with a fixed cleanup instruction."""
        raw = self._run(
            SUBTITLE_PROMPT,
            guidance_mode="v2v_apg",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            video=input.fileKey,
        )
        return SubtitleRemoveOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.method()
    @node_slot(NodeSlots.REMOVE_WATERMARK)
    def remove_watermark(self, input: RemoveWatermarkInput) -> RemoveWatermarkOutput:
        """Watermark removal — v2v with a fixed cleanup instruction."""
        raw = self._run(
            WATERMARK_PROMPT,
            guidance_mode="v2v_apg",
            num_frames=VIDEO_FRAMES,
            out_suffix=".mp4",
            video=input.fileKey,
        )
        return RemoveWatermarkOutput(success=True, video=asset(raw, mime="video/mp4"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

