"""Visualisation. Enough to see whether the physics is behaving, no more."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .state import OBS_LAYERS, FireState

_IDX = {name: i for i, name in enumerate(OBS_LAYERS)}

# Unburnt fuel, burnt ground, flame, water, retardant.
_GREEN = np.array([0.22, 0.42, 0.20])
_BLACK = np.array([0.13, 0.11, 0.10])
_FLAME = np.array([1.00, 0.45, 0.05])
_WATER = np.array([0.25, 0.55, 0.95])
_RETARDANT = np.array([0.85, 0.35, 0.45])


def composite(layers: Tensor, index: int = 0) -> np.ndarray:
    """`[H, W, 3]` RGB view of one world from a stacked `[B, C, H, W]` tensor."""
    x = layers[index].detach().cpu().numpy()
    fuel = x[_IDX["fuel"]]
    burned = x[_IDX["burned"]]
    intensity = x[_IDX["intensity"]]
    water = x[_IDX["water"]]
    retardant = x[_IDX["retardant"]]

    img = _GREEN * fuel[..., None] + _BLACK * np.clip(burned, 0, 1)[..., None]
    img = _blend(img, _WATER, np.clip(water * 1.5, 0, 0.8))
    img = _blend(img, _RETARDANT, np.clip(retardant, 0, 0.85))
    img = _blend(img, _FLAME, np.clip(intensity, 0, 1))
    return np.clip(img, 0, 1)


def _blend(base: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return base * (1 - a) + color * a


def draw_platforms(
    img: np.ndarray,
    positions: Tensor,
    load_fraction: Tensor | None = None,
    upscale: int = 1,
    size: int = 2,
    index: int = 0,
) -> np.ndarray:
    """Mark each vehicle on an RGB `uint8` frame, in place.

    Without this you only see a platform when it drops, which makes the
    reload cycle invisible - the fleet appears to teleport between splashes.
    Markers are drawn as crosses on a dark halo so they stay legible over
    flame, and dimmed in proportion to remaining load, so a vehicle fades as
    it empties and brightens when it refills.

    `positions` is `[B, N, 2]` in (y, x) cells; `upscale` must match the zoom
    already applied to `img`.
    """
    pos = positions[index].detach().cpu().numpy()
    if load_fraction is None:
        loads = np.ones(len(pos))
    else:
        loads = load_fraction[index].detach().cpu().numpy().clip(0.0, 1.0)

    h, w = img.shape[:2]
    for (y, x), load in zip(pos, loads):
        cy, cx = int(round(float(y) * upscale)), int(round(float(x) * upscale))
        brightness = 0.35 + 0.65 * float(load)
        for radius, colour in ((size + 1, np.zeros(3)), (size, np.full(3, 255 * brightness))):
            ys = slice(max(cy - radius, 0), min(cy + radius + 1, h))
            xs = slice(max(cx - radius, 0), min(cx + radius + 1, w))
            if 0 <= cx < w:
                img[ys, cx] = colour
            if 0 <= cy < h:
                img[cy, xs] = colour
    return img


def save_png(state: FireState, path: str | Path, index: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(composite(state.stack(), index), origin="lower")
    ax.set_title(f"t = {state.t[index].item() / 60:.0f} min   "
                 f"burned = {state.burned_area()[index].item():.1f} ha")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_gif(frames: list[Tensor], path: str | Path, index: int = 0, stride: int = 4, fps: int = 12) -> Path:
    """Animate a `rollout(record=True)` frame list."""
    import imageio.v2 as imageio

    path = Path(path)
    imgs = [(composite(f, index) * 255).astype(np.uint8) for f in frames[::stride]]
    # imageio >= 2.28 dropped `fps` for the pillow plugin; `duration` is in ms.
    imageio.mimsave(path, imgs, duration=1000.0 / fps, loop=0)
    return path


def panel(states: dict[str, FireState], path: str | Path, index: int = 0) -> Path:
    """Side-by-side comparison of named scenarios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, axes = plt.subplots(1, len(states), figsize=(4.2 * len(states), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, st) in zip(axes, states.items()):
        ax.imshow(composite(st.stack(), index), origin="lower")
        ax.set_title(f"{name}\n{st.burned_area()[index].item():.1f} ha burned", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
