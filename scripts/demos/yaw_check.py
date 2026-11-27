"""Does the hand turn about the vertical when the teleop client's Q / E keys say so? Sends the same orientation targets as the client, without a PC, and measures.

Client, runs in .venv, AFTER the env server is up with the task to test (ONE environment, the gripper closed on the object as for the teleoperation):

    bash jobs/env_server.sh --set num_envs=1 --set gripper_can_open=false          # ENV_CONFIG=configs/task/nut_thread_forge.yaml for another task
    python -m scripts.demos.yaw_check --out logs/yaw_check_nut_thread
    # or in a Slurm job, server included:   ENV_CONFIG=configs/task/nut_thread_forge.yaml sbatch jobs/job_demos.sh yaw_check --out logs/yaw_check_nut_thread

What it does: from the held orientation of the reset it turns the target by `--rate-deg` per step (the client's `--yaw-rate-deg`, 3 degrees) up to `--total-deg`, then
turns back to 0, the hand's position target unchanged. Each step is the client's: the IK target of the server's `ik_joint_target`, the action = measured joints +
clip(IK target - measured, +-max_joint_step), gripper closed (the same arithmetic as src/envs/backend.py `_teleop_step`). Printed every few steps and written to
<out>/yaw_check.json, with yaw_check.mp4 (the frames of the videos, banner = the numbers):

    yaw measured   the rotation of the fingertip about the vertical since the reset (degrees, counter-clockwise seen from above): it must follow the command, with the SAME sign
    tilt           how far the hand's axis left its direction (degrees): ~0, the rotation must stay about the vertical
    hand drift     how far the fingertip moved (mm): ~0 (the position target is unchanged)
    object drift   how far the held object (peg - socket offset, xy) moved (mm): 0 if the object turns about its own axis, ~ (distance of its axis to the fingertip) x angle if it turns about the fingertip
    JOINT CLIP     steps where a joint target was held back by max_joint_step (0.05 rad): the wrist joint turns by about the yaw angle, so 3 degrees (0.052 rad) per step is at the limit
    held           the taxels (robot with taxels) still show contact on both fingers at the end, and the object offset did not jump: the object was not lost
The verdict lines at the end say what to look at; nothing is asserted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.demos.single_env import SingleEnv
from src.eval.video_panels import add_banner, compose_frame, panel_kwargs, tactile_stats


def quat_mul(a, b):
    """Hamilton product a * b of two quaternions (w, x, y, z): b first, then a, both in the same fixed frame (the same as scripts/demos/teleop_client.py)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def yaw_quat(angle):
    """Rotation of `angle` radians about the vertical (env z), counter-clockwise seen from above (w, x, y, z)."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotation_since(q0, q):
    """(yaw, tilt) in degrees of the rotation that took the orientation q0 to q, in the fixed frame: yaw = its component about the vertical (swing-twist about z), tilt = what is left."""
    rel = quat_mul(np.asarray(q, dtype=np.float64), quat_conj(np.asarray(q0, dtype=np.float64)))
    rel = rel / np.linalg.norm(rel)
    if rel[0] < 0:
        rel = -rel
    twist = np.array([rel[0], 0.0, 0.0, rel[3]])
    norm = np.linalg.norm(twist)
    if norm < 1e-9:                                      # a half turn about a horizontal axis: all tilt
        return 0.0, 180.0
    twist = twist / norm
    yaw = 2.0 * np.arctan2(twist[3], twist[0])
    swing = quat_mul(rel, quat_conj(twist))
    tilt = 2.0 * np.arctan2(np.linalg.norm(swing[1:]), abs(swing[0]))
    return float(np.rad2deg(yaw)), float(np.rad2deg(tilt))


def schedule(total_deg, rate_deg):
    """The yaw (degrees) the target has at each step: 0 -> total by `rate`, then back to 0 by `rate`."""
    n = int(round(abs(total_deg) / abs(rate_deg)))
    up = [rate_deg * np.sign(total_deg) * (i + 1) for i in range(n)]
    return up + up[::-1][1:] + [0.0]


def main(argv=None):
    p = argparse.ArgumentParser(description="Turn the hand about the vertical as the teleop client does and measure it.")
    p.add_argument("--host", default="localhost", help="Env server (see the top of the file).")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--out", default="logs/yaw_check", help="Output folder (yaw_check.json, yaw_check.mp4, one png per quarter).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate-deg", type=float, default=3.0, help="Yaw added to the target per step (the client's --yaw-rate-deg).")
    p.add_argument("--total-deg", type=float, default=60.0, help="Yaw of the turn out before it comes back (negative: the other way).")
    p.add_argument("--settle-steps", type=int, default=5, help="Steps with the target held at the end of each way, before it is measured.")
    p.add_argument("--max-joint-step", type=float, default=0.05, help="The client's safety clip on a joint target (rad per step).")
    p.add_argument("--cmd-gain", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args(argv)

    import imageio.v3 as iio
    env = SingleEnv(args.host, args.port, gripper_can_open=False)
    debug = env.debug
    info = debug("robot_info")
    task = info.get("task", "(not told by the server)")
    print(f"task {task}   prompt {env.prompt!r}   robot {info['name']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs = env.reset(args.seed)
    pose0 = debug("fingertip_pose")
    quat0, pos0 = np.asarray(pose0["quat"], dtype=np.float64), np.asarray(pose0["pos"], dtype=np.float64)
    offset0 = np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)
    rows, frames, clips = [], [], 0
    plan = schedule(args.total_deg, args.rate_deg)
    # the target is held `settle_steps` more steps at the top and at the end, to read where the hand really went
    top = int(np.argmax(np.abs(plan)))
    steps = [(y, False) for y in plan[:top + 1]] + [(plan[top], True)] * args.settle_steps + [(y, False) for y in plan[top + 1:]] + [(plan[-1], True)] * args.settle_steps

    for i, (yaw_cmd, settle) in enumerate(steps, start=1):
        target = quat_mul(yaw_quat(np.deg2rad(yaw_cmd)), quat0)
        ik = debug("ik_joint_target", delta_pos=np.zeros((1, 3), np.float32), target_quat=target.astype(np.float32).reshape(1, 4))
        q_now, q_ik = np.asarray(ik["q_now"], dtype=np.float64).reshape(-1), np.asarray(ik["q_target"], dtype=np.float64).reshape(-1)
        dq = args.cmd_gain * (q_ik - q_now)
        clipped = bool(np.abs(dq).max() > args.max_joint_step)
        clips += int(clipped)
        env.step(np.concatenate([q_now + np.clip(dq, -args.max_joint_step, args.max_joint_step), [1.0]]).astype(np.float32))
        obs = env.observation()
        pose = debug("fingertip_pose")
        yaw, tilt = rotation_since(quat0, pose["quat"])
        hand_mm = 1000.0 * float(np.linalg.norm(np.asarray(pose["pos"], dtype=np.float64) - pos0))
        offset = np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)
        obj_mm = 1000.0 * float(np.linalg.norm(offset[:2] - offset0[:2]))
        stats = tactile_stats(obs.get("extra/tactile"))
        st = env.status()
        rows.append({"step": i, "yaw_cmd": float(yaw_cmd), "yaw": yaw, "tilt": tilt, "hand_mm": hand_mm, "object_mm": obj_mm, "clipped": clipped,
                     "joint7_dq": float(dq[6]), "force_n": float(np.linalg.norm(np.asarray(obs["extra/ft_force"], dtype=np.float64))),
                     "tactile_max_n": None if stats is None else stats[0], "settle": settle, "ended": bool(st.done)})
        text = [(f"yaw cmd {yaw_cmd:+6.1f}  measured {yaw:+6.1f} deg   tilt {tilt:4.1f} deg   step {i}/{len(steps)}", (255, 255, 255)),
                (f"hand drift {hand_mm:5.2f} mm   object drift {obj_mm:5.2f} mm   joint 7 dq {dq[6]:+.3f} rad" + ("   JOINT CLIP" if clipped else ""), (200, 200, 200)),
                (f"hand force {rows[-1]['force_n']:5.2f} N" + ("" if stats is None else f"   taxel max {stats[0]:5.2f} N"), (120, 180, 255))]
        frames.append(add_banner(compose_frame(obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"], **panel_kwargs(obs)), text))
        if i == 1 or i % 5 == 0 or i == len(steps):
            r = rows[-1]
            print(f"step {i:3d}  cmd {r['yaw_cmd']:+6.1f}  yaw {r['yaw']:+7.2f}  tilt {r['tilt']:5.2f}  hand {r['hand_mm']:5.2f} mm  object {r['object_mm']:5.2f} mm  "
                  f"force {r['force_n']:5.2f} N" + ("  CLIP" if clipped else ""), flush=True)
        if st.done:
            print(f"the episode ended at step {i} ({'success' if st.success else 'timeout'}): the rest is not measured")
            break

    iio.imwrite(out / "yaw_check.mp4", np.stack(frames), fps=args.fps, codec="libx264")
    for name, idx in (("start", 0), ("top", top + args.settle_steps), ("end", len(frames) - 1)):
        iio.imwrite(out / f"yaw_{name}.png", frames[min(idx, len(frames) - 1)])
    at_top = [r for r in rows if r["settle"] and abs(r["yaw_cmd"]) > 0]
    at_end = rows[-1]
    top_row = at_top[-1] if at_top else rows[-1]
    summary = {"task": task, "seed": args.seed, "rate_deg": args.rate_deg, "total_deg": args.total_deg, "steps": rows, "clipped_steps": clips,
               "top": {k: top_row[k] for k in ("yaw_cmd", "yaw", "tilt", "hand_mm", "object_mm")}, "end": {k: at_end[k] for k in ("yaw", "tilt", "hand_mm", "object_mm")}}
    (out / "yaw_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out}/: yaw_check.mp4 yaw_start.png yaw_top.png yaw_end.png yaw_check.json")
    t = summary["top"]
    follows = abs(t["yaw"] - t["yaw_cmd"]) <= max(2.0, 0.1 * abs(t["yaw_cmd"])) and np.sign(t["yaw"]) == np.sign(t["yaw_cmd"])
    print(f"at the top: commanded {t['yaw_cmd']:+.1f} deg, measured {t['yaw']:+.1f} deg  -> " + ("the hand follows, with the right sign (Q = counter-clockwise seen from above)" if follows
          else "CHECK: it does not follow (sign, joint limit, or the clip: see JOINT CLIP steps)"))
    print(f"tilt {t['tilt']:.1f} deg ({'vertical axis kept' if t['tilt'] < 3.0 else 'CHECK: the hand tilted'}), hand drift {t['hand_mm']:.1f} mm, "
          f"object drift {t['object_mm']:.1f} mm ({t['object_mm'] / max(abs(t['yaw']), 1e-6):.2f} mm per degree: ~0 = turns about its own axis, larger = about the fingertip)")
    print(f"back at 0: yaw {summary['end']['yaw']:+.1f} deg, object drift {summary['end']['object_mm']:.1f} mm   joint clips: {clips} of {len(rows)} steps"
          + ("   (lower --rate-deg or raise --max-joint-step if the clip is what stops the hand)" if clips else ""))
    if at_end["tactile_max_n"] is not None:
        print(f"taxels at the end: max {at_end['tactile_max_n']:.2f} N ({'still held' if at_end['tactile_max_n'] > 0.05 else 'CHECK: no contact, the object may have been lost'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
