# worker-ltx-video

[![Runpod](https://api.runpod.io/badge/alithabit1/worker-ltx-video)](https://console.runpod.io/hub/alithabit1/worker-ltx-video)

Custom RunPod Serverless worker for **Lightricks LTX-Video**, the fastest
open-source video diffusion model in 2026 (~10–20 seconds per 5 s clip on an
RTX 4090).

Why this exists: the hub templates for LTX on RunPod either require Blackwell
GPUs (RTX 5090 / B200, scarce) or wrap ComfyUI workflow JSON (verbose API).
This worker uses HuggingFace `diffusers` directly, runs on any 24 GB+ GPU
(RTX 4090, A6000, L40S, …), and exposes a tiny API.

**Files:**
- `rp_handler.py` — RunPod serverless handler (T2V + I2V via diffusers)
- `Dockerfile` — container build instructions
- `requirements.txt` — Python deps
- `.runpod/hub.json` — RunPod Hub deployment config

## API

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

```json
{
  "input": {
    "prompt": "a red panda walking through a sunlit forest",
    "negative_prompt": "blurry, low quality",
    "image_base64": "<optional — provide for I2V mode>",
    "width": 704,
    "height": 480,
    "num_frames": 121,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "fps": 24,
    "seed": 42
  }
}
```

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | **Required** |
| `image_base64` | — | If present → I2V. If absent → pure T2V. |
| `width` / `height` | 704 / 480 | Rounded down to a multiple of 32. |
| `num_frames` | 121 | ~5 s at 24 fps. |
| `num_inference_steps` | 40 | Lower = faster, less quality (try 25–30). |
| `guidance_scale` | 3.0 | Lower = looser, higher = stricter. |
| `seed` | 42 | Change for variation. |

Response (when `COMPLETED`):

```json
{
  "output": {
    "video_base64": "<MP4 bytes, base64-encoded>",
    "width": 704, "height": 480, "num_frames": 121, "fps": 24,
    "seed": 42, "mode": "t2v"
  }
}
```

## Deploying

This repo is hub-ready (`.runpod/hub.json`). The fastest path is:

1. Push this repo to GitHub (public).
2. In RunPod console → **Hub → Add your repo** → paste the GitHub URL.
3. RunPod builds the image and exposes a one-click Deploy button.
4. After deploy, grab the **Endpoint ID** from the endpoint page.

Alternative (manual Docker):

```bash
docker build -t <you>/worker-ltx-video:v1 .
docker push <you>/worker-ltx-video:v1
# then create a Serverless endpoint pointing at <you>/worker-ltx-video:v1
```

## GPU recommendation

24 GB Pro (RTX 4090 / A6000) is the sweet spot. The pipeline uses
bf16 + VAE slicing/tiling so it fits comfortably.

## Cold start

First boot per worker pulls the model weights (~13 GB) from HuggingFace —
expect 90–180 s. Subsequent calls within the idle window reuse the warm
worker and finish in 10–30 s.
