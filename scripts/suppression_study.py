"""What the fire does on its own, and what a platform actually does to it.

Four modes, all sharing one scenario definition so their numbers compare:

    freeburn  the unopposed baseline every controller must beat. Steps the
              propagator directly with no platform attached, which is the only
              way to get a genuinely suppressant-free run - see the note on
              `zero_action` below.
    sweep     burned area against `dispatch_s`, the delay between ignition and
              the first legal drop.
    drops     a log of every release: when, how much, where, and where the fire
              head was at the time. This is how you check that drop frequency
              is the reload timer and drop position tracks the front.
    gif       animation plus a six-panel time-lapse strip.

    python scripts/suppression_study.py drops --dispatch 3600
    python scripts/suppression_study.py sweep --steps 2500

Note on baselines: `env.zero_action()` is not a no-release action. `Platform.step`
clamps the action to [-1, 1] before `intent = sigmoid(4 * a)`, so the lowest
reachable intent is sigmoid(-4) = 1.8% of capacity per step and the tank keeps
draining. `freeburn` therefore bypasses the platform entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import head_seeking_policy  # noqa: E402

from swarmfire import EnvConfig, FireEnv, render  # noqa: E402

DROP_THRESHOLD_L = 1.0  # a load delta below this is numerical dust, not a drop


def make_env(args, dispatch_s: float = 0.0, platform: str | None = None) -> FireEnv:
    torch.manual_seed(args.seed)
    cfg = EnvConfig(
        batch=1,
        grid=(args.grid, args.grid),
        horizon=args.steps,
        wind=tuple(args.wind),
        preset=args.preset,
        device=args.device,
        dt=args.dt,
        dispatch_s=dispatch_s,
    )
    env = FireEnv(platform or args.platform, config=cfg)
    env.reset()
    return env


def head_of(state, index: int = 0) -> tuple[int, int] | None:
    """Cell of peak intensity, or None once the fire is out."""
    intensity = state.intensity[index]
    if intensity.max() <= 1e-4:
        return None
    return divmod(int(intensity.argmax().item()), state.grid[1])


def frame(state, upscale: int = 1, platform=None) -> np.ndarray:
    img = (render.composite(state.stack()) * 255).astype(np.uint8)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    if platform is not None:
        render.draw_platforms(
            img,
            platform.position,
            load_fraction=platform.load / platform.spec.capacity_l,
            upscale=upscale,
            size=max(2, 2 * upscale),
        )
    # `strip` draws with origin="lower"; flip so the animation agrees with it.
    return np.flipud(img).copy()


def run(env, args, record_every: int = 0, upscale: int = 1):
    """Roll out under the scripted policy, logging every release.

    Returns (drops, frames, curve). A drop is detected from the drop in tank
    load rather than from the action, so it counts what was actually delivered.
    """
    policy = head_seeking_policy(env, args.lead)
    obs = env.observe()
    drops, frames, curve = [], [], []

    for i in range(args.steps):
        load_before = env.platform.load.clone()
        position = env.platform.position.clone().flatten().tolist()
        head, burned = head_of(env.state), env.state.burned_area()[0].item()
        minutes = env.state.t[0].item() / 60.0

        obs, *_ = env.step(policy(obs))

        volume = (load_before - env.platform.load).flatten()[0].item()
        if volume > DROP_THRESHOLD_L:
            drops.append(dict(minutes=minutes, volume=volume, position=position,
                              head=head, burned=burned))
        if record_every and i % record_every == 0:
            frames.append(frame(env.state, upscale, env.platform))
            curve.append((env.state.t[0].item() / 3600.0, env.state.burned_area()[0].item()))

    return drops, frames, curve


# ----------------------------------------------------------------- the modes


def mode_freeburn(args):
    """No platform at all: propagator only, so nothing can be deposited."""
    env = make_env(args)
    state = env.state
    frames, curve = [], []
    for i in range(args.steps):
        state = env.propagator.step(state, args.dt).detach()
        if i % args.record_every == 0:
            frames.append(frame(state))
            curve.append((state.t[0].item() / 3600.0, state.burned_area()[0].item()))

    assert state.water.max().item() == 0.0, "suppressant leaked into a free burn"
    print(f"free burn: {curve[-1][1]:.1f} ha after {curve[-1][0]:.2f} h")

    strip(frames, curve, args.out / "timelapse_truefree.png",
          f"true free burn - {args.preset}, wind {tuple(args.wind)} m/s")
    save_gif(frames, args.out / "true_free_burn.gif", args.fps)


def mode_sweep(args):
    header = f"{'dispatch':>10} {'1st drop':>10} {'ha then':>9} {'final ha':>9} {'drops':>6} {'kL':>7}"
    print(header)
    print("-" * len(header))
    for minutes in args.dispatch_sweep:
        env = make_env(args, dispatch_s=minutes * 60.0)
        drops, _, _ = run(env, args)
        final = env.state.burned_area()[0].item()
        if drops:
            first = f"{drops[0]['minutes']:.0f} min"
            then = f"{drops[0]['burned']:.1f}"
        else:
            first, then = "never", "-"
        total = sum(d["volume"] for d in drops) / 1000.0
        print(f"{minutes:8.0f} m {first:>10} {then:>9} {final:9.1f} {len(drops):6d} {total:7.1f}")


def mode_drops(args):
    env = make_env(args, dispatch_s=args.dispatch)
    drops, _, _ = run(env, args)
    if not drops:
        print("no drops - dispatch delay longer than the horizon?")
        return

    print(f"{'t (min)':>8} {'volume L':>9} {'drop (y,x)':>15} {'head (y,x)':>12} {'burned ha':>10}")
    for d in drops:
        head = f"({d['head'][0]:4d},{d['head'][1]:4d})" if d["head"] else "     out    "
        pos = f"({d['position'][0]:6.1f},{d['position'][1]:6.1f})"
        print(f"{d['minutes']:8.0f} {d['volume']:9.0f} {pos:>15} {head:>12} {d['burned']:10.1f}")

    gaps = [b["minutes"] - a["minutes"] for a, b in zip(drops, drops[1:])]
    spec = env.platform.spec
    print(f"\n{len(drops)} drops, {sum(d['volume'] for d in drops) / 1000:.0f} kL total")
    if gaps:
        print(f"gap between drops: {min(gaps):.1f} - {max(gaps):.1f} min "
              f"(reload_s = {spec.reload_s / 60:.0f} min)")
    print(f"final: {env.state.burned_area()[0].item():.1f} ha")


def mode_gif(args):
    env = make_env(args, dispatch_s=args.dispatch)
    drops, frames, curve = run(env, args, record_every=args.record_every, upscale=args.upscale)
    tag = f"{args.platform}_dispatch_{int(args.dispatch // 60)}min"
    save_gif(frames, args.out / f"{tag}.gif", args.fps)
    strip(frames, curve, args.out / f"{tag}.png",
          f"{args.platform}, dispatch {args.dispatch / 60:.0f} min, "
          f"reload {env.platform.spec.reload_s / 60:.0f} min")
    print(f"{len(drops)} drops, final {env.state.burned_area()[0].item():.1f} ha")


def mode_compare(args):
    """One animation, several scenarios side by side, sharing an ignition."""
    runs = []

    env = make_env(args)
    state = env.state
    frames, curve = [], []
    for i in range(args.steps):
        state = env.propagator.step(state, args.dt).detach()
        if i % args.record_every == 0:
            frames.append(frame(state, args.upscale))
            curve.append((state.t[0].item() / 3600.0, state.burned_area()[0].item()))
    runs.append(("free burn", frames, curve))
    print(f"free burn: {curve[-1][1]:.1f} ha")

    for name in args.compare:
        env = make_env(args, dispatch_s=args.dispatch, platform=name)
        drops, frames, curve = run(env, args, record_every=args.record_every, upscale=args.upscale)
        runs.append((name.replace("_", " "), frames, curve))
        print(f"{name}: {len(drops)} drops, {curve[-1][1]:.1f} ha")

    side_by_side(runs, args.out / "comparison.gif", args.fps)


# ------------------------------------------------------------------- output


def side_by_side(runs, path: Path, fps: int, gap: int = 10, font_size: int = 20) -> None:
    """Stitch several equally-sized frame sequences into one labelled strip."""
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    font_file = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_file), font_size)
    small = ImageFont.truetype(str(font_file), int(font_size * 0.8))

    n = min(len(frames) for _, frames, _ in runs)
    h, w = runs[0][1][0].shape[:2]
    header = int(font_size * 2.6)
    canvas_w = len(runs) * w + (len(runs) - 1) * gap

    out = []
    for i in range(n):
        canvas = Image.new("RGB", (canvas_w, h + header), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for k, (name, frames, curve) in enumerate(runs):
            x0 = k * (w + gap)
            canvas.paste(Image.fromarray(frames[i]), (x0, header))
            draw.text((x0 + 4, 2), name, fill=(15, 15, 15), font=font)
            draw.text((x0 + 4, 4 + font_size), f"{curve[i][0]:.1f} h    {curve[i][1]:.0f} ha",
                      fill=(90, 90, 90), font=small)
        out.append(np.asarray(canvas))

    import imageio.v2 as imageio

    imageio.mimsave(path, out, duration=1000.0 / fps, loop=0)
    print(f"  wrote {path} ({len(out)} frames, {canvas_w}x{h + header})")


def save_gif(frames, path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    # imageio >= 2.28 dropped `fps` for the pillow plugin; `duration` is in ms.
    imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)
    print(f"  wrote {path}")


def strip(frames, curve, path: Path, title: str, panels: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = np.linspace(0, len(frames) - 1, panels).astype(int)
    fig, axes = plt.subplots(1, panels, figsize=(3.2 * panels, 3.8))
    for ax, k in zip(axes, picks):
        # Frames arrive already flipped for the animation; undo it here so the
        # panels keep the origin="lower" convention used elsewhere in render.py.
        ax.imshow(np.flipud(frames[k]), origin="lower", interpolation="nearest")
        ax.set_title(f"t = {curve[k][0]:.1f} h\n{curve[k][1]:.0f} ha", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    # High enough that a 3x-upscaled frame is not resampled below its own
    # resolution, which would erase the platform markers.
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeburn", "sweep", "drops", "gif", "compare"])
    ap.add_argument("--platform", default="helicopter")
    ap.add_argument("--compare", nargs="+", default=["helicopter", "drone_swarm"],
                    help="compare mode: platforms to show beside the free burn")
    ap.add_argument("--dispatch", type=float, default=3600.0, help="dispatch delay, seconds")
    ap.add_argument("--dispatch-sweep", type=float, nargs="+", default=[0, 10, 30, 60, 120],
                    help="sweep mode: dispatch delays in minutes")
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--wind", type=float, nargs=2, default=[5.0, 2.0])
    ap.add_argument("--preset", default="grass")
    ap.add_argument("--lead", type=float, default=0.0, help="cells to aim downwind of the head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-every", type=int, default=12, help="steps between recorded frames")
    ap.add_argument("--upscale", type=int, default=3, help="nearest-neighbour zoom for the gif")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    {"freeburn": mode_freeburn, "sweep": mode_sweep, "drops": mode_drops,
     "gif": mode_gif, "compare": mode_compare}[args.mode](args)


if __name__ == "__main__":
    main()
