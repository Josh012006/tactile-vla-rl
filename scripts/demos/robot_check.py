"""Look at the simulated robot: reset, hold, open, close the gripper, with pictures, a video and the taxel readings.

Client, runs in .venv, AFTER the env server is up with the robot to test (the server decides which robot it runs: the YAML
field `robot`, or `--set robot=...` on the server command line), with ONE environment. The gripper must be allowed to open:

    bash jobs/env_server.sh --set num_envs=1 --set robot=robotiq_2f85_tsf --set gripper_can_open=true
    python -m scripts.demos.robot_check --out reports/robot_check_tsf
    # or in a Slurm job, server included:   sbatch jobs/job_demos.sh robot_check --out reports/robot_check_tsf

Phases (the arm is held where it is, only the gripper command changes): hold (the peg is in the hand), open (command 0),
close (command 1), hold. For each phase it prints the gripper observation, the finger opening (m per finger), the distance
between the two finger bodies, the contact force of the hand and the taxels (largest zone, number of zones in contact per
finger). Written to --out: one png per phase (the 448 x 448 frame of the videos) and robot_check.mp4 (448 x 496, with the
banner). What to look at:
    - the wrist camera shows the fingers at the bottom of the image, the peg between them (hold),
    - open: the finger distance grows and the peg falls; close: it shrinks again,
    - taxels (robotiq_2f85_tsf): non-zero on both fingers while the peg is held, zero when the gripper is open.
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from scripts.demos.single_env import SingleEnv
from src.eval.video_panels import add_banner, compose_frame, panel_kwargs, tactile_stats

GRIPPER_OPEN_WIDTH = 0.04   # m per finger (src/envs/isaaclab/forge_env.py; not imported: it needs Isaac Lab)

p = argparse.ArgumentParser(description="Reset, hold, open and close the gripper of the simulated robot and look at it.")
p.add_argument("--host", default="localhost", help="Env server (see the top of the file).")
p.add_argument("--port", type=int, default=8102)
p.add_argument("--out", default="reports/robot_check", help="Output folder (png per phase + robot_check.mp4).")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hold-steps", type=int, default=10)
p.add_argument("--open-steps", type=int, default=15)
p.add_argument("--close-steps", type=int, default=15)
p.add_argument("--fps", type=int, default=8)
args = p.parse_args()

env = SingleEnv(args.host, args.port)
debug = env.debug


info = debug("robot_info")
print(f"robot: {info['name']} - {info['description']}")
print(f"gripper: {info['gripper']}   taxels: {info['tactile']}   gripper_can_open: {info['gripper_can_open']}")
print("joints:", info["joint_names"])
if info["gripper_adapter"]:
    print("adapter:", info["gripper_adapter"])
can_open = bool(info["gripper_can_open"])
if not can_open:
    print("NOTE: gripper_can_open is false on this server -> only the 'hold' phase is run "
          "(restart it with --set gripper_can_open=true).")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
obs = env.reset(args.seed)
frames, last = [], {}


def snapshot(phase, t):
    ft = np.asarray(obs["extra/ft_force"], dtype=np.float64)
    tac = obs.get("extra/tactile")
    stats = tactile_stats(tac)
    width, dist = float(debug("finger_width")), float(debug("finger_distance"))
    zones = None if tac is None else [int((np.asarray(tac)[s] > 0.05).sum()) for s in (0, 1)]
    last[phase] = {"width": width, "dist": dist, "tac": stats, "zones": zones}
    frame = compose_frame(obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"], **panel_kwargs(obs))
    text = [(f"{phase:5s} t={t:3d}  gripper obs {float(np.ravel(obs['observation/gripper_position'])[0]):.3f}", (255, 255, 255)),
            (f"finger width {1000 * width:5.2f} mm  distance {1000 * dist:6.2f} mm", (200, 200, 200)),
            (f"hand force {np.linalg.norm(ft):5.2f} N" + ("" if stats is None else f"  taxel max {stats[0]:5.2f} N"), (120, 180, 255))]
    frames.append(add_banner(frame, text))
    return frame, width, dist, stats, zones, ft


def run(phase, steps, command):
    """command None = hold the gripper where it is."""
    global obs
    for t in range(steps):
        q = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)[:7]
        cmd = command if command is not None else float(np.clip(1.0 - float(debug("finger_width")) / GRIPPER_OPEN_WIDTH, 0, 1))
        env.step(np.concatenate([q, [cmd]]).astype(np.float32))
        obs = env.observation()
        frame, width, dist, stats, zones, ft = snapshot(phase, t + 1)
        if t == 0 or t == steps - 1:
            print(f"{phase:5s} t={t + 1:3d}  obs gripper {float(np.ravel(obs['observation/gripper_position'])[0]):6.3f}  "
                  f"width {1000 * width:6.2f} mm  distance {1000 * dist:7.2f} mm  hand force {np.linalg.norm(ft):6.2f} N"
                  + ("" if stats is None else f"  taxel max {stats[0]:6.2f} N (L {stats[1]:.2f} / R {stats[2]:.2f}), zones > 0.05 N: L {zones[0]} R {zones[1]}"))
    iio.imwrite(out / f"{phase}.png", frame)


frame0, *_ = snapshot("reset", 0)
iio.imwrite(out / "reset.png", frame0)
print(f"reset t=  0  distance {1000 * last['reset']['dist']:7.2f} mm  width {1000 * last['reset']['width']:.2f} mm")
run("hold", args.hold_steps, None)
if can_open:
    run("open", args.open_steps, 0.0)
    run("close", args.close_steps, 1.0)
    run("hold2", args.hold_steps, None)
iio.imwrite(out / "robot_check.mp4", np.stack(frames), fps=args.fps, codec="libx264")
print(f"\nwritten to {out}/: reset.png hold.png" + (" open.png close.png hold2.png" if can_open else "") + " robot_check.mp4")
if can_open:
    d = {k: 1000 * v["dist"] for k, v in last.items()}
    print(f"finger distance: hold {d['hold']:.1f} mm -> open {d['open']:.1f} mm -> close {d['close']:.1f} mm "
          f"({'opens and closes' if d['open'] > d['hold'] + 5 and d['close'] < d['open'] - 5 else 'CHECK: it did not open / close as expected'})")
if info["tactile"]:
    held = last["hold"]["tac"]
    print(f"taxels while holding: max {held[0]:.2f} N" + (f", open: {last['open']['tac'][0]:.2f} N" if can_open else "")
          + ("  -> contact seen on both fingers" if last["hold"]["zones"][0] and last["hold"]["zones"][1] else "  -> CHECK: no contact on one finger"))
