"""Frames and video of a rollout: exterior | wrist cameras with a text bar (step, plan, peg-socket distance, force). numpy + Pillow;
imageio (with its ffmpeg, a dependency of lerobot) only to write the mp4, a GIF otherwise."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.envs import protocol as P

SCALE = 2                   # the cameras are 224 x 224: shown at 448 x 448 each
BAR = 32                    # text bar under the two images (pixels)


def compose_frame(obs: Dict[str, np.ndarray], env: int, t: int, plan_no: int, sample: Optional[dict], success: bool) -> np.ndarray:
    """exterior | wrist, enlarged, with a text bar: step, plan number, peg-socket xy / z, force, success. (H, W, 3) uint8, sides multiples of 16."""
    from PIL import Image, ImageDraw, ImageFont

    def big(img):
        return np.repeat(np.repeat(np.asarray(img, dtype=np.uint8), SCALE, axis=0), SCALE, axis=1)

    ext, wrist = big(obs[P.EXTERIOR_IMAGE][env]), big(obs[P.WRIST_IMAGE][env])
    body = np.concatenate([ext, wrist], axis=1)
    frame = np.zeros((body.shape[0] + BAR, body.shape[1], 3), np.uint8)
    frame[:body.shape[0]] = body
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=16)                    # Pillow >= 10.1; older versions only have the small bitmap font
    except TypeError:
        font = ImageFont.load_default()

    def text(xy, s, fill=(255, 255, 255), box=False):
        if box:                                                    # readable over any image
            x0, y0, x1, y1 = d.textbbox(xy, s, font=font)
            d.rectangle((x0 - 3, y0 - 2, x1 + 3, y1 + 2), fill=(0, 0, 0))
        d.text(xy, s, fill=fill, font=font)

    text((6, 4), "exterior", box=True)
    text((ext.shape[1] + 6, 4), "wrist", box=True)
    bar = f"t={t:3d}  plan {plan_no:2d}"
    if sample is not None:
        bar += f"   peg-socket xy {sample['xy_mm']:6.1f} mm  z {sample['z_mm']:6.1f} mm   |F| {sample['force_n']:5.1f} N"
    text((6, body.shape[0] + 6), bar)
    if success:
        text((body.shape[1] - 100, body.shape[0] + 6), "SUCCESS", fill=(80, 255, 80))
    frame = np.asarray(im)
    pad_h, pad_w = (-frame.shape[0]) % 16, (-frame.shape[1]) % 16                 # h264 wants macroblocks
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0))) if pad_h or pad_w else frame


def write_video(frames: List[np.ndarray], path: Path, fps: int = 15) -> str:
    """mp4 (h264) through imageio-ffmpeg; if that is not available, an animated GIF next to it. Returns the path written."""
    path = Path(path)
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=7)
        try:
            for f in frames:
                writer.append_data(f)
        finally:
            writer.close()
        return str(path)
    except Exception as e:  # noqa: BLE001
        from PIL import Image
        gif = path.with_suffix(".gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        print(f"[video] mp4 not written ({e!r}); wrote {gif} instead")
        return str(gif)
