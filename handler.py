"""RunPod serverless handler for LTX-Video text-to-video and image-to-video.

Loads the LTX-Video pipeline once at cold start, then serves requests through
the /run endpoint. Supports two modes:

- T2V: send {"input": {"prompt": "..."}} — pure text-to-video.
- I2V: send {"input": {"prompt": "...", "image_base64": "..."}} — animate an image.

Returns: {"output": {"video_base64": "<mp4 bytes, base64>", ...}}
"""

import base64
import io
import os
import tempfile
import traceback

import numpy as np
import torch
from PIL import Image

import runpod
from diffusers import LTXImageToVideoPipeline, LTXPipeline
from diffusers.utils import export_to_video


MODEL_ID = os.environ.get("LTX_MODEL_ID", "Lightricks/LTX-Video")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, (value // multiple) * multiple)


print(f"[init] loading LTX-Video pipelines (model={MODEL_ID}, dtype={DTYPE}, device={DEVICE})")

t2v_pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
i2v_pipe = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

# Memory savings — without these, 24 GB cards can OOM at 768x512.
for pipe in (t2v_pipe, i2v_pipe):
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

print("[init] pipelines ready")


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
            print(f"[i2v] {width}x{height} frames={num_frames} steps={steps}")
            image = _decode_image(image_b64).resize((width, height))
            result = i2v_pipe(
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
            print(f"[t2v] {width}x{height} frames={num_frames} steps={steps}")
            result = t2v_pipe(
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
        print("[handler] error:", e)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
