"""RunPod serverless handler for LTX-Video text-to-video and image-to-video.

CRITICAL: this worker REFUSES to run on CPU. RunPod sometimes starts the
Python process before the GPU driver is fully ready; the previous version
silently fell back to CPU and one inference took 12+ minutes before timing
out. We now wait for CUDA explicitly and assert it stayed available.

Pipelines are lazy-loaded per-request — only T2V is initialised when an I2V
request arrives, and vice versa. That halves cold-start VRAM use vs pre-loading
both.

Supports:

- T2V: send {"input": {"prompt": "..."}} - pure text-to-video.
- I2V: send {"input": {"prompt": "...", "image_base64": "..."}} - animate.

Returns: {"output": {"video_base64": "<mp4, base64>", ...}}
"""

import base64
import io
import os
import tempfile
import time
import traceback

import numpy as np
import torch
from PIL import Image

import runpod
from diffusers import LTXImageToVideoPipeline, LTXPipeline
from diffusers.utils import export_to_video


MODEL_ID = os.environ.get("LTX_MODEL_ID", "Lightricks/LTX-Video")


def _wait_for_cuda(max_seconds: int = 90) -> None:
    """Block until torch sees a CUDA device. Fail loudly if it never does."""
    deadline = time.monotonic() + max_seconds
    while not torch.cuda.is_available():
        if time.monotonic() > deadline:
            raise RuntimeError(
                "CUDA never became available — refusing to run on CPU. "
                "Check that the endpoint actually requested a GPU worker."
            )
        time.sleep(1)
    # Exercise a tiny GPU allocation so we fail fast if the device is broken.
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()


print("[init] waiting for CUDA…", flush=True)
_wait_for_cuda()
DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"[init] CUDA ready: {torch.cuda.get_device_name(0)}  dtype={DTYPE}", flush=True)


_PIPES: dict[str, object] = {}


def _get_pipe(kind: str):
    """Lazy-load and cache one pipeline. kind ∈ {'t2v', 'i2v'}."""
    if kind in _PIPES:
        return _PIPES[kind]
    cls = LTXPipeline if kind == "t2v" else LTXImageToVideoPipeline
    print(f"[init] loading {kind} pipeline …", flush=True)
    pipe = cls.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    # Hard verify the pipeline actually landed on the GPU.
    sample_tensor = next(pipe.transformer.parameters(), None) \
        if hasattr(pipe, "transformer") else None
    if sample_tensor is not None and not sample_tensor.is_cuda:
        raise RuntimeError(
            f"{kind} pipeline did not end up on CUDA after .to('cuda') — "
            f"device is {sample_tensor.device}. Refusing to continue on CPU."
        )
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _PIPES[kind] = pipe
    print(f"[init] {kind} pipeline ready on {DEVICE}", flush=True)
    return pipe


def _round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, (value // multiple) * multiple)


def _decode_image(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _frames_to_mp4_b64(frames: list, fps: int) -> str:
    np_frames = [np.array(f) for f in frames]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    try:
        export_to_video(np_frames, out_path, fps=fps)
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def handler(event: dict) -> dict:
    try:
        # Defensive: re-check CUDA at request time too.
        if not torch.cuda.is_available():
            return {"error": "CUDA disappeared between init and request — worker is unhealthy"}

        inputs = event.get("input") or {}
        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return {"error": "input.prompt is required"}

        negative_prompt = inputs.get("negative_prompt") or (
            "worst quality, blurry, distorted, watermark, low resolution"
        )
        width = _round_to_multiple(int(inputs.get("width", 704)))
        height = _round_to_multiple(int(inputs.get("height", 480)))
        num_frames = int(inputs.get("num_frames", 121))
        steps = int(inputs.get("num_inference_steps", 40))
        guidance = float(inputs.get("guidance_scale", 3.0))
        fps = int(inputs.get("fps", 24))
        seed = int(inputs.get("seed", 42))

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        image_b64 = inputs.get("image_base64") or inputs.get("image")
        if image_b64:
            print(f"[i2v] {width}x{height} frames={num_frames} steps={steps}", flush=True)
            pipe = _get_pipe("i2v")
            image = _decode_image(image_b64).resize((width, height))
            result = pipe(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        else:
            print(f"[t2v] {width}x{height} frames={num_frames} steps={steps}", flush=True)
            pipe = _get_pipe("t2v")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        frames = result.frames[0]
        video_b64 = _frames_to_mp4_b64(frames, fps=fps)

        return {
            "video_base64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "mode": "i2v" if image_b64 else "t2v",
        }
    except Exception as e:
        print("[handler] error:", e, flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
