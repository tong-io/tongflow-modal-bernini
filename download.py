"""Modal download entry for tongflow-modal-bernini.

Run:
  modal run download.py::download

Fetches the self-contained Bernini-R 1.3B diffusers directory (Wan base
components + Bernini-R transformer / transformer_2 weights) into the shared
`models` volume. Self-contained: no local imports.
"""

from __future__ import annotations

import os

import modal

REPO_ID = "ByteDance/Bernini-R-1.3B-Diffusers"

volume = modal.Volume.from_name("models", create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub>=0.34.0,<1.0"
    ),
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    local_dir = f"/models/{REPO_ID}"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=local_dir,
        token=token,
        local_dir_use_symlinks=False,
    )
    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
