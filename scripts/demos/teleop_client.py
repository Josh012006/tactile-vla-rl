"""Teleoperation client for the FORGE simulations (PegInsert, GearMesh, NutThread) -- STEP 3/3: you drive the arm and record demos.

Runs on YOUR PC (Windows / WSL is fine), not on the cluster. The simulation keeps running in the env server (src/envs/isaaclab/server.py) on an
L40S node (jobs/env_server.sh); this script connects to it through an SSH tunnel and shows what the policy sees:

    left        exterior camera (224x224, enlarged)
    middle      wrist camera    (224x224, enlarged)
    far right   the TAXEL PANEL of the demo / replay videos (robot with taxels, i.e. robot=robotiq_2f85_tsf on the
                server): the two 7x4 grids of contact force (row 0 = fingertip at the bottom), the colour scale
                (0 .. --tactile-vmax N, same as tactile_vmax of the YAML), the verdict (BOTH FINGERS / ONE FINGER
                ONLY / NO CONTACT), zones max / sum, zones in contact, hand force, gripper effort. A robot without
                taxels gets the same placeholder as in the videos. --no-tactile removes the column.
    under       the offset peg -> hole (ground truth, simulation only), the command you sent, the contact force,
                the state of the inputs and the time of every round trip to the server

You move the held object (the peg, the gear or the nut; "peg" below means that object: it is rigidly held, the fingertip follows) with the mouse and
the keyboard. The task is the server's (its YAML, configs/task/*_forge.yaml): the client reads its name and its prompt from the server and sends nothing
task-specific. The orientation of the hand is held fixed, exactly as in the scripted demos, unless you turn it about the vertical with Q / E (NutThread:
the nut has to be screwed on; the other tasks never need it), and the gripper stays closed. Each control step the
server turns your move into 7 joint targets (FORGE's own IK), applies them and answers with the new observation:
one round trip per step (debug op "teleop_step" of src/envs/backend.py).

THE SERVER must be started with ONE environment, the 5-step success rule of the scripted demos and a closed gripper:

    bash jobs/env_server.sh --set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false
    (or: sbatch jobs/job_teleop_server.sh, which starts it on an L40S and prints the node and the two commands to run on your PC)

The client checks all three at start-up and says what to change.

SLOW MOTION. The simulation only advances when you step it, and one step is always 1/15 s of simulated time, however
long it takes on the wall clock (~0.3-0.4 s through the tunnel, the simulation alone needs ~0.18 s). By default the
simulation steps ONLY when you give an input (mouse move, W/S or Space), so you can stop and look at no cost. What
is recorded is therefore at the policy's 15 Hz, in the same format as the scripted demos. C switches to AUTO mode,
where the simulation steps at every loop even without input (needed to let the 5 "inserted" steps pass once the peg
is in the hole).

RECORDING. Every step is kept in memory: the observation the move was computed from, and the 8 values sent to the arm
(7 absolute joint targets + gripper 1.0), exactly like collect_waypoint_demos.py. An episode ends 5 steps in a row
after FORGE says "inserted" (same rule as the scripted demos), or at the step limit. When it succeeded, press Enter:
the whole episode is sent to the env server, which writes it ON THE CLUSTER, in --out-dir (relative to the project
root there, default demos/isaaclab/Forge<Task>_Robotiq_teleop with <Task> from the server's task (ForgePegInsert, ForgeGearMesh, ForgeNutThread), in the
`demos` link to scratch):

    ep_0000_seed100.npz   exterior_image, wrist_image, joint_position, gripper_position, ft_force,
                          peg_socket_offset, actions, prompt, seed  (same keys as the scripted demos), plus
                          tactile (T, 2, 7, 4) and gripper_effort (T,) when the server's robot has taxels
    index.jsonl           one line per saved demo: seed, steps, wall seconds, final offset, max force, ...

then the next seed starts. Not saved: failed episodes, and successes you skip with R / Z (nothing is written
until you press Enter). The numbering ep_NNNN continues after the files already in the folder, so you can quit and
come back; use another --seed for the next session (the same seed gives the same start state).

This file is self-contained on purpose: it imports nothing from the repo (the wire format of src/envs/wire.py is copied below). Install (PowerShell, or WSL):

    py -m venv teleop-venv
    teleop-venv\\Scripts\\activate
    pip install numpy pygame websockets msgpack

Tunnel (second terminal, leave it open; <node> = the node of your salloc, e.g. cn-l042):

    ssh -J mila -L 8102:localhost:8102 <node>

Run (this one file is all you need on the PC; copy scripts/demos/teleop_client.py there):

    python teleop_client.py --host localhost --port 8102 --seed 100

Controls (Tab first: the mouse must be captured, otherwise nothing moves):
    mouse        move the peg in the horizontal plane, as seen in the WRIST camera (right image): mouse up = towards
                 the top of that image, mouse right = towards its right (the hand goes where you point; the scene
                 then slides the other way in the image). --view exterior uses the left image instead.
                 No button to hold: once captured, every movement counts. The pixels travelled since the previous
                 step are added up and sent as ONE move (a swipe = one step, capped at --max-xy-mm).
    arrow keys   the same directions: Up / Down / Left / Right = towards the top / bottom / left / right of the wrist
                 image. --key-xy-mm per step, repeated while the key is down; they add up with the mouse.
    W / S        peg up / down
    Q / E        turn the hand about the vertical, --yaw-rate-deg per step while the key is down (Q = counter-clockwise seen from above, E = clockwise).
                 The wrist-image directions of the mouse and the arrows turn with the hand. Needed for NutThread, harmless elsewhere (never pressed = fixed
                 orientation, as before).
    Shift        fine mode   (x 0.25: for the last millimetres)
    Ctrl         fast mode   (x 2.5)
    Space        one step with no move (let the arm settle)
    C            AUTO steps on / off (the simulation steps at every loop, also without input)
    Enter        after a SUCCESS: save the demo on the cluster and start the next seed
    Z            same seed again (discards the current episode: retry the same start state)
    R            next seed (discards the current episode)
    Tab          capture / release the mouse      Esc  quit
The caps per step (--max-xy-mm, --max-z-mm) are the same order as the scripted demos; the size of a move is
--xy-gain-mm per mouse pixel and --z-rate-mm per step with W / S. If a direction is wrong for your window, use
--flip-x / --flip-y / --rotate-deg (the directions are printed at every reset).
"""
import argparse
import collections
import dataclasses
import io
import statistics
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.exceptions
import websockets.sync.client


# ---- msgpack + numpy: same wire format as src/envs/wire.py (copied, so no repo needed) ----
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject or obj.dtype.kind in "VOUS":
            raise TypeError(f"cannot serialise an array of dtype {obj.dtype}")
        return {"__ndarray__": True, "data": np.ascontiguousarray(obj).tobytes(), "dtype": obj.dtype.str, "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return {"__npscalar__": True, "data": obj.item(), "dtype": obj.dtype.str}
    raise TypeError(f"cannot serialise {type(obj)}")


def _unpack_array(obj):
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if "__npscalar__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def _pack(message):
    return msgpack.packb(message, default=_pack_array, use_bin_type=True)


def _unpackb(raw):
    return msgpack.unpackb(raw, object_hook=_unpack_array, raw=False)


# ---- Constants ------------------------------------------------------------------------------------
# Exterior camera pose of src/envs/isaaclab/forge_env.py (EXTERIOR_EYE / EXTERIOR_TARGET, env frame, m):
# only used to turn "mouse right / forward" into directions of the table plane as seen in the left image.
EXTERIOR_EYE = (1.05, 0.35, 0.35)
EXTERIOR_TARGET = (0.57, 0.02, 0.08)
# Wrist camera orientation in the fingertip frame (quaternion w x y z, ROS convention: image right = camera x, image
# down = camera y, optical axis = camera z). This is the Franka Hand camera; the server tells the one of its own robot
# (debug op "robot_info", see Session.start), this value is only the fallback for a server that does not know the op.
WRIST_ROT = (0.6916, -0.1471, -0.1471, 0.6916)
HOLD_AFTER_SUCCESS = 5      # same as the scripted demos: the episode ends this many steps after "inserted"
# observation key -> array name in the .npz (same as RECORDED of collect_waypoint_demos.py)
RECORDED = {
    "observation/exterior_image_1_left": "exterior_image",
    "observation/wrist_image_left": "wrist_image",
    "observation/joint_position": "joint_position",
    "observation/gripper_position": "gripper_position",
    "extra/ft_force": "ft_force",
    "extra/peg_socket_offset": "peg_socket_offset",
}
# saved only when the server sends it (robot with taxels): (T, 2, 7, 4) N, [:, 0] left finger, [:, 1] right
RECORDED_OPTIONAL = {"extra/tactile": "tactile", "extra/gripper_effort": "gripper_effort"}


# ---- Connection to the env server ----------------------------------------------------------------
class RemoteEnv:
    """Client of src/envs/ws_server.py (ops spec / reset / step / debug), with ONE environment, plus a timer on every call."""

    def __init__(self, host, port):
        self.uri = f"ws://{host}:{port}"
        self.timings = collections.defaultdict(lambda: collections.deque(maxlen=200))   # name -> ms
        self._connect()
        self.spec = self._call("spec", "spec")["spec"]

    def _connect(self):
        # ping_interval=None: no keepalive. Saving a demo sends ~10-30 MB through the tunnel in one message, which can
        # take longer than the default 20 s pong deadline and then closes the connection (seen on the cluster).
        kw = dict(compression=None, max_size=None, open_timeout=15, close_timeout=100)
        try:
            try:
                self.conn = websockets.sync.client.connect(self.uri, ping_interval=None, **kw)
            except TypeError:          # old websockets (< 15): no keepalive at all, nothing to disable
                self.conn = websockets.sync.client.connect(self.uri, **kw)
        except Exception as e:     # noqa: BLE001 -- any failure here means "no server behind the tunnel"
            raise SystemExit(
                f"cannot connect to {self.uri}: {type(e).__name__}: {e}\n"
                "  - is the ssh tunnel still open?   ssh -J mila -L <port>:localhost:<port> <node>\n"
                "  - is the env server running on that node (and finished loading Isaac Sim: '[env_server] ready')?") from e

    def reconnect(self):
        """New connection to the same server: the simulation (and the episode in progress) is untouched."""
        self.close()
        self._connect()

    def _call(self, name, op, **request):
        t0 = time.perf_counter()
        self.conn.send(_pack({"op": op, **request}))
        response = _unpackb(self.conn.recv())
        self.timings[name].append((time.perf_counter() - t0) * 1000.0)
        if response.get("status") != "ok":
            raise RuntimeError(f"server error in {op}: {response.get('message')}")
        return response

    def check_server(self):
        """The server must run ONE environment, end an episode after HOLD_AFTER_SUCCESS 'inserted' steps in a row and keep the gripper closed."""
        spec, problems = self.spec, []
        if spec["num_envs"] != 1:
            problems.append(f"it runs {spec['num_envs']} environments: --set num_envs=1")
        if spec["success_hold_steps"] != HOLD_AFTER_SUCCESS:
            problems.append(f"its success_hold_steps is {spec['success_hold_steps']}, not {HOLD_AFTER_SUCCESS} (as in the scripted demos): "
                            f"--set success_hold_steps={HOLD_AFTER_SUCCESS}")
        if spec["robot"].get("gripper_can_open", True):
            problems.append("its gripper can open (the demos keep it closed on the peg): --set gripper_can_open=false")
        if problems:
            raise SystemExit("the env server is not set up for teleoperation:\n  - " + "\n  - ".join(problems)
                             + "\nRestart it:  bash jobs/env_server.sh --set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false")

    @property
    def prompt(self):
        return self.spec["prompt"]

    @staticmethod
    def _row0(x):
        """The row of environment 0: arrays (1, ...) -> (...), dicts element by element."""
        if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == 1:
            return x[0]
        if isinstance(x, dict):
            return {k: RemoteEnv._row0(v) for k, v in x.items()}
        return x

    def reset(self, seed):
        """Round 0 of `seed`: the first observation of the episode ({key: array of the one environment})."""
        return {k: np.array(v[0]) for k, v in self._call("reset", "reset", seed=int(seed))["obs"].items()}

    def debug(self, kind, timer=None, **kw):
        result = self._row0(self._call(timer or f"debug:{kind}", "debug", kind=kind, **kw)["result"])
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            hint = "  (is the env server an old version? pull and restart it)" if "unknown debug kind" in result["error"] else ""
            raise RuntimeError(result["error"] + hint)
        return result

    def teleop_step(self, delta_pos, target_quat, cmd_gain, max_joint_step):
        """One control step in ONE round trip: IK + step + observation + flags (ManagerBackend._teleop_step). Returns
        obs (the observation after the step, {key: array}), action (8,), clipped, and the step's flags."""
        r = self._call("teleop_step", "debug", kind="teleop_step", delta_pos=np.asarray(delta_pos, dtype=np.float32).reshape(1, 3),
                       target_quat=np.asarray(target_quat, dtype=np.float32).reshape(1, 4), cmd_gain=float(cmd_gain),
                       max_joint_step=float(max_joint_step))["result"]
        if isinstance(r, dict) and "error" in r and len(r) == 1:
            raise RuntimeError(r["error"])
        return {"obs": {k: np.array(v[0]) for k, v in r["obs"].items()}, "action": np.asarray(r["action"])[0],
                "clipped": bool(r["clipped"][0]), "inserted": bool(r["inserted"][0]), "terminated": bool(r["terminated"][0]),
                "truncated": bool(r["truncated"][0])}

    def save_demo(self, out_dir, npz_bytes, seed, info, success):
        """The server writes the .npz (already built, compressed) in out_dir (relative to the project root there)."""
        return self.debug("save_teleop_demo", timer="save", out_dir=str(out_dir), npz=npz_bytes,
                          seed=int(seed), info=info, success=bool(success))

    def measure_rtt(self, n=20):
        """Round trip of a call that does nothing on the server: the network floor of every call."""
        for _ in range(n):
            self.debug("task_geometry", timer="rtt")

    def close(self):
        try:
            self.conn.close()
        except Exception:      # noqa: BLE001 -- already closed
            pass

    def mean_ms(self, name):
        values = self.timings.get(name)
        return statistics.fmean(values) if values else float("nan")

    def p95_ms(self, name):
        values = sorted(self.timings.get(name, ()))
        return values[int(0.95 * (len(values) - 1))] if values else float("nan")


# ---- Inputs -> move -------------------------------------------------------------------------------
@dataclasses.dataclass
class Inputs:
    """What the operator did since the previous step."""
    dx: int = 0              # mouse pixels, summed since the previous step (right +)
    dy: int = 0              # mouse pixels, summed since the previous step (down +)
    up: bool = False         # W
    down: bool = False       # S
    fine: bool = False       # Shift
    fast: bool = False       # Ctrl
    wait: bool = False       # Space: one step without moving
    left: bool = False       # arrow keys: horizontal moves in the directions of the exterior image
    right: bool = False
    forward: bool = False    # Up arrow: away from the camera
    back: bool = False       # Down arrow: towards the camera
    yaw_left: bool = False   # Q: turn the hand counter-clockwise about the vertical (seen from above)
    yaw_right: bool = False  # E: clockwise

    def any(self):
        return bool(self.dx or self.dy or self.up or self.down or self.wait
                    or self.left or self.right or self.forward or self.back
                    or self.yaw_left or self.yaw_right)


def screen_axes(flip_x=False, flip_y=False, rotate_deg=0.0):
    """Unit vectors (x, y of the env frame) of 'mouse right' and 'mouse forward' for the exterior camera.

    forward = the horizontal direction the camera looks along (away from the camera); right = to the right of the
    image (forward x up). The robot base is at the env origin; the camera looks at the table from the +x side.
    """
    f = np.asarray(EXTERIOR_TARGET[:2], dtype=np.float64) - np.asarray(EXTERIOR_EYE[:2], dtype=np.float64)
    forward = f / np.linalg.norm(f)
    right = np.array([forward[1], -forward[0]])
    a = np.deg2rad(rotate_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    right, forward = rot @ right, rot @ forward
    if flip_x:
        right = -right
    if flip_y:
        forward = -forward
    return right, forward


def quat_to_matrix(q):
    """3x3 rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    """Hamilton product a * b of two quaternions (w, x, y, z): b first, then a, both in the same fixed frame."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def yaw_quat(angle):
    """Rotation of `angle` radians about the vertical (env z), counter-clockwise seen from above, as a quaternion (w, x, y, z)."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def default_out_dir(task):
    """Where the demos of a task go on the cluster: Forge<Task>_Robotiq_teleop (peg_insert -> ForgePegInsert, gear_mesh -> ForgeGearMesh, nut_thread -> ForgeNutThread).
    A server that does not say its task (older than this client) is the PegInsert one."""
    camel = "".join(part.capitalize() for part in str(task or "peg_insert").split("_"))
    return f"demos/isaaclab/Forge{camel}_Robotiq_teleop"


def wrist_axes(hand_quat, flip_x=False, flip_y=False, rotate_deg=0.0, wrist_rot=WRIST_ROT):
    """Unit vectors (x, y of the env frame) of 'towards the right' and 'towards the top' of the WRIST image.

    The camera is fixed on the hand, whose orientation is held (hand_quat = fingertip quaternion of the server,
    w x y z): image right / up are the camera's x / -y axes in the env frame, projected on the horizontal plane.
    The hand looks down, so the camera is nearly vertical (tilted ~24 deg) and these projections are well defined.
    """
    rot = quat_to_matrix(hand_quat) @ quat_to_matrix(wrist_rot)
    right, up = rot[:2, 0], -rot[:2, 1]
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    a = np.deg2rad(rotate_deg)
    r2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    right, up = r2 @ right, r2 @ up
    return (-right if flip_x else right), (-up if flip_y else up)


def view_axes(args, hand_quat, wrist_rot=WRIST_ROT):
    """Directions of mouse right / mouse up in the env frame, for the chosen view (wrist or exterior image)."""
    if args.view == "exterior":
        return screen_axes(args.flip_x, args.flip_y, args.rotate_deg)
    return wrist_axes(hand_quat, args.flip_x, args.flip_y, args.rotate_deg, wrist_rot)


def build_delta(inp, axes, xy_gain_mm, z_rate_mm, max_xy_mm, max_z_mm, key_xy_mm=2.0):
    """The peg translation (m, env frame) of one control step, from the inputs. Capped per step."""
    scale = 0.25 if inp.fine else (2.5 if inp.fast else 1.0)
    right, forward = axes
    # mouse (pygame's y grows downwards) + arrow keys, both in the directions of the exterior image
    xy = xy_gain_mm * scale * (inp.dx * right + (-inp.dy) * forward)
    xy = xy + key_xy_mm * scale * ((int(inp.right) - int(inp.left)) * right + (int(inp.forward) - int(inp.back)) * forward)
    norm = float(np.linalg.norm(xy))
    if norm > max_xy_mm:
        xy = xy * (max_xy_mm / norm)
    z = float(np.clip((int(inp.up) - int(inp.down)) * z_rate_mm * scale, -max_z_mm, max_z_mm))
    return np.array([xy[0], xy[1], z]) * 1e-3


def build_yaw(inp, yaw_rate_deg):
    """The rotation (rad) about the vertical of one control step, from Q (+) / E (-); both down = none. Fine / fast scale it as they scale the moves."""
    scale = 0.25 if inp.fine else (2.5 if inp.fast else 1.0)
    return float(np.deg2rad(yaw_rate_deg * scale * (int(inp.yaw_left) - int(inp.yaw_right))))


# ---- One teleoperated episode ---------------------------------------------------------------------
class Session:
    """State of the current episode. No pygame in here (so it can be tested without a screen)."""

    def __init__(self, env, args):
        self.env, self.args = env, args
        self.seed = None
        self.obs = None
        self.quat0 = None
        self.yaw = 0.0                 # rad turned about the vertical since the start of the episode (Q / E)
        self.prompt = ""

    def start(self, seed):
        self.seed = seed
        self.obs = self.env.reset(seed)
        self.quat0 = self.env.debug("fingertip_pose")["quat"]       # orientation held for the whole episode
        if getattr(self, "wrist_rot", None) is None:                # camera of the server's robot (asked once)
            robot = self.env.debug("robot_info")
            rot = robot.get("wrist_rot_ros_in_fingertip") if isinstance(robot, dict) else None
            self.wrist_rot = tuple(rot) if rot else WRIST_ROT
            self.robot_name = robot.get("name", "?") if isinstance(robot, dict) else "?"
        self.yaw = 0.0
        self.axes = view_axes(self.args, self.quat0, self.wrist_rot)   # mouse right / up in the env frame
        self.step_count = 0
        self.success_run = 0
        self.success = False
        self.timed_out = False
        self.over = False
        self.clipped_steps = 0
        self.last_delta = np.zeros(3)
        self.last_clipped = False
        self.recorded = {**RECORDED, **{k: v for k, v in RECORDED_OPTIONAL.items() if k in self.obs}}
        self.frames = {name: [] for name in self.recorded.values()}   # what is saved: the observations ...
        self.actions = []                                           # ... and the actions sent
        self.idle_steps = 0                                         # steps without any move
        self.max_force = 0.0
        self.t_first = None                                         # wall clock of the first / last step
        self.t_last = None
        self.saved = False

    def step(self, delta, dyaw=0.0):
        """One control step: move the held object by `delta` (m, env frame) and turn the hand by `dyaw` (rad, about the vertical)."""
        if self.t_first is None:
            self.t_first = time.perf_counter()
        # the observation the action is computed from (the one on screen), like the scripted demos
        for key, name in self.recorded.items():
            self.frames[name].append(np.asarray(self.obs[key]))
        if dyaw:
            self.yaw += float(dyaw)
        target_quat = quat_mul(yaw_quat(self.yaw), self.quat0) if self.yaw else self.quat0       # the orientation held, turned by the yaw so far
        r = self.env.teleop_step(delta, target_quat, self.args.cmd_gain, self.args.max_joint_step)
        self.t_last = time.perf_counter()
        self.actions.append(np.asarray(r["action"], dtype=np.float32))
        if dyaw and self.args.view == "wrist":                      # the wrist camera turned with the hand: so do the directions of the mouse
            self.axes = view_axes(self.args, target_quat, self.wrist_rot)
        self.idle_steps += int(not (np.any(delta) or dyaw))
        self.max_force = max(self.max_force, float(np.linalg.norm(np.asarray(self.obs["extra/ft_force"], dtype=np.float64))))
        self.last_delta, self.last_clipped = np.asarray(delta, dtype=np.float64), bool(r["clipped"])
        self.clipped_steps += int(self.last_clipped)
        self.step_count += 1
        self.obs = r["obs"]          # the server never auto-resets here: after the last step it is still this episode's real last observation
        self.success_run = self.success_run + 1 if r["inserted"] else 0
        if r["terminated"] or self.success_run >= HOLD_AFTER_SUCCESS:
            self.success, self.over = True, True
        elif r["truncated"]:
            self.timed_out, self.over = True, True
        if self.step_count >= self.args.max_steps:
            self.over = True

    def arrays(self):
        out = {name: np.stack(values) for name, values in self.frames.items()}
        out["actions"] = np.stack(self.actions)
        return out

    def info(self):
        """One line of index.jsonl (plain python types: it is sent to the server and written as JSON)."""
        final = np.asarray(self.obs["extra/peg_socket_offset"], dtype=np.float64) * 1000.0
        return {"seed": int(self.seed), "success": bool(self.success), "teleop": True, "steps": len(self.actions),
                "idle_steps": int(self.idle_steps), "clipped_steps": int(self.clipped_steps),
                "wall_seconds": round(float(self.t_last - self.t_first), 1) if self.t_first else 0.0,
                "final_offset_mm": [round(float(v), 2) for v in final], "max_force_N": round(self.max_force, 2),
                "yaw_deg": round(float(np.rad2deg(self.yaw)), 1)}

    def save(self):
        """Send the finished episode to the server, which writes it on the cluster. True when it worked."""
        if self.saved or not self.success or not self.args.out_dir:
            return False
        # the .npz is built here (losslessly compressed: 3-5x smaller to send) and written as is by the server
        t0 = time.perf_counter()
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.arrays(), prompt=np.array(self.prompt), seed=np.array(self.seed))
        data = buf.getvalue()
        print(f"sending the demo ({len(self.actions)} steps, {len(data) / 1e6:.1f} MB) to the cluster ...", flush=True)
        r = None
        for attempt in (1, 2):
            try:
                r = self.env.save_demo(self.args.out_dir, data, self.seed, self.info(), True)
                break
            except websockets.exceptions.ConnectionClosed as e:     # lost the tunnel / server: reconnect, once
                print(f"   connection lost ({type(e).__name__}), reconnecting ...", flush=True)
                try:
                    self.env.reconnect()
                except Exception as e2:     # noqa: BLE001
                    print(f"   cannot reconnect: {e2}")
                    break
            except Exception as e:     # noqa: BLE001 -- refused by the server (bad out_dir, ...), nothing to retry
                print(f"!! NOT saved: {e}")
                break
        if r is None:
            # never lose a demo: keep a copy next to the client (copy it to the cluster by hand, e.g. with scp)
            backup = Path(f"teleop_unsaved_seed{self.seed}_{time.strftime('%H%M%S')}.npz")
            backup.write_bytes(data)
            print(f"!! NOT saved on the cluster -- a copy is in {backup.resolve()} ; Enter tries again")
            return False
        print(f"   ({time.perf_counter() - t0:.1f} s)")
        self.saved = True
        print(f"saved {r['file']}  ({len(self.actions)} steps)  ->  {r['path']}   [{r['count']} demos in the folder]")
        return True

    def offset_mm(self):
        return np.asarray(self.obs["extra/peg_socket_offset"], dtype=np.float64) * 1000.0

    def force_n(self):
        return float(np.linalg.norm(np.asarray(self.obs["extra/ft_force"], dtype=np.float64)))


# ---- Display --------------------------------------------------------------------------------------
BACKGROUND = (18, 18, 22)
TEXT = (230, 230, 230)
DIM = (140, 140, 150)
GOOD = (90, 200, 120)
WARN = (240, 190, 70)
BAD = (235, 90, 90)


def to_surface(pygame, image, scale):
    """HxWx3 uint8 array -> pygame surface, enlarged by an integer factor (nearest neighbour)."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8).swapaxes(0, 1))   # pygame wants (W, H, 3)
    surface = pygame.surfarray.make_surface(array)
    return pygame.transform.scale(surface, (array.shape[0] * scale, array.shape[1] * scale))


def draw_text(screen, font, x, y, text, colour=TEXT):
    screen.blit(font.render(text, True, colour), (x, y))
    return y + font.get_linesize()


# ---- Taxel panel: the one of the demo / replay videos (src/eval/video_panels.py), redrawn here with pygame ----
# Same layout, same colours, same readings. This file imports nothing from the repo on purpose, hence the copy.
# The panel is 2:1 (448 x 224 px in the videos); every length is multiplied by height / 224 so that it can be enlarged.
TACTILE_CONTACT_N = 0.05      # a zone above this force counts as "in contact"
_CMAP_KEYS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_CMAP_COLORS = np.array([(18, 18, 40), (40, 80, 220), (40, 200, 120), (240, 220, 50), (230, 40, 30)], dtype=np.float64)
_FONT_CACHE = {}


def colormap(x):
    """x in [0, 1] (any shape) -> uint8 colours (..., 3): dark blue (no contact) .. green .. yellow .. red."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.stack([np.interp(x, _CMAP_KEYS, _CMAP_COLORS[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


def sized_font(pygame, size):
    size = int(max(8, size))
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("consolas,dejavusansmono,couriernew", size)
    return _FONT_CACHE[size]


def tactile_from_obs(obs):
    """The (2, rows, cols) taxel array of an observation, or None (robot without taxels / unexpected shape)."""
    t = obs.get("extra/tactile")
    if t is None:
        return None
    t = np.asarray(t, dtype=np.float64)
    return t if t.ndim == 3 and t.shape[0] == 2 else None


def tactile_panel_surface(pygame, tactile, width, height, vmax, hand_force=None, gripper_effort=None):
    """pygame surface (width x height) of the taxel panel. `tactile` None -> the placeholder of the videos."""
    k = height / 224.0
    img = np.full((height, width, 3), 12, dtype=np.uint8)
    if tactile is None:
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        label = sized_font(pygame, round(13 * k)).render("no tactile data (robot without taxels)", True, (130, 130, 130))
        surface.blit(label, ((width - label.get_width()) // 2, (height - label.get_height()) // 2))
        return surface
    t = np.asarray(tactile, dtype=np.float64)
    n_rows, n_cols = t.shape[1], t.shape[2]
    cell = max(8, (height - round(70 * k)) // n_rows)               # 22 px for a 224 px panel
    gw, gh = n_cols * cell, n_rows * cell
    gap, x_first, top = round(18 * k), round(14 * k), round(40 * k)
    x_grid = [x_first, x_first + gw + gap]
    for side in (0, 1):
        colours = colormap(t[side] / max(vmax, 1e-9))               # (rows, cols, 3)
        for r in range(n_rows):
            y1 = top + gh - r * cell                                # row 0 (fingertip) at the bottom
            for c in range(n_cols):
                img[y1 - cell:y1, x_grid[side] + c * cell:x_grid[side] + (c + 1) * cell] = colours[r, c]
        img[top - 1, x_grid[side]:x_grid[side] + gw] = img[top + gh, x_grid[side]:x_grid[side] + gw] = 70   # frame
    bar_x, bar_w = x_grid[1] + gw + round(10 * k), max(8, round(12 * k))     # colour bar, as tall as the grids
    img[top:top + gh, bar_x:bar_x + bar_w] = colormap(np.linspace(1.0, 0.0, gh))[:, None, :]            # vmax at the top
    img[top - 1, bar_x:bar_x + bar_w] = img[top + gh, bar_x:bar_x + bar_w] = 70
    surface = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))

    big, small = sized_font(pygame, round(15 * k)), sized_font(pygame, round(13 * k))
    for side, name in enumerate(("LEFT", "RIGHT")):
        label = big.render(name, True, (255, 255, 255))
        surface.blit(label, (x_grid[side] + (gw - label.get_width()) // 2, round(14 * k)))
    in_contact = [int((t[i] > TACTILE_CONTACT_N).sum()) for i in (0, 1)]
    if all(in_contact):
        state, colour = "BOTH FINGERS", (120, 230, 140)
    elif any(in_contact):
        state, colour = "ONE FINGER ONLY", (255, 190, 90)
    else:
        state, colour = "NO CONTACT", (150, 150, 150)
    lines = [(state, colour),
             (f"zones max  L {t[0].max():4.1f}  R {t[1].max():4.1f} N", (230, 230, 230)),
             (f"zones sum  L {t[0].sum():4.1f}  R {t[1].sum():4.1f} N", (230, 230, 230)),
             (f"in contact L {in_contact[0]}/{n_rows * n_cols}  R {in_contact[1]}/{n_rows * n_cols}", (230, 230, 230))]
    if hand_force is not None:
        lines.append((f"hand force  {hand_force:5.2f} N", (230, 230, 230)))
    if gripper_effort is not None:
        lines.append((f"gripper effort  {gripper_effort:5.1f} Nm", (230, 230, 230)))
    blue = (130, 150, 255)
    x_lab = bar_x + bar_w + round(4 * k)
    surface.blit(small.render("N", True, blue), (bar_x - 2, top - round(20 * k)))
    tick_w = 0
    for frac in (0.0, 0.5, 1.0):                                    # ticks: 0, vmax / 2, vmax (colours saturate above vmax)
        y = top + round(gh * (1.0 - frac))
        pygame.draw.line(surface, blue, (bar_x + bar_w, y), (bar_x + bar_w + round(3 * k), y))
        tick = small.render(f"{vmax * frac:g}", True, blue)
        tick_w = max(tick_w, tick.get_width())
        surface.blit(tick, (x_lab, y - tick.get_height() // 2))
    x_text = x_lab + tick_w + round(26 * k)
    avail = width - x_text - 4
    size = round(13 * k)                                            # shrink the readings if they would leave the panel
    while size > 8 and max(sized_font(pygame, size).size(text)[0] for text, _ in lines) > avail:
        size -= 1
    font, step = sized_font(pygame, size), round(22 * k)
    for i, (text, col) in enumerate(lines):
        surface.blit(font.render(text, True, col), (x_text, round(44 * k) + i * step))
    return surface


def main():
    p = argparse.ArgumentParser(description="Teleop client, step 3: drive the arm with the mouse and the keyboard, "
                                            "record demos on the cluster.")
    p.add_argument("--host", default="localhost", help="Local end of the ssh tunnel.")
    p.add_argument("--port", type=int, default=8102, help="Local port of the tunnel (server_port of the YAML, 8102 by default).")
    p.add_argument("--seed", type=int, default=100,
                   help="Reset seed of the first episode; the next episode (R, or after a save) uses seed + 1, ... "
                        "Use another value for each session: the same seed gives the same start state.")
    p.add_argument("--out-dir", default=None,
                   help="Folder ON THE CLUSTER where the server writes the demos, relative to the project root there "
                        "(the folder that holds src/; `demos` is the link to scratch). Must stay inside the project. '' = record nothing. "
                        "Default: demos/isaaclab/Forge<Task>_Robotiq_teleop, <Task> being the server's task (ForgePegInsert, ForgeGearMesh, ForgeNutThread).")
    p.add_argument("--soft-limit-steps", type=int, default=90,
                   help="The step counter turns orange beyond this many steps (the policy has 150; shorter demos "
                        "are closer to the scripted ones). Display only.")
    p.add_argument("--scale", type=int, default=3, help="Camera enlargement factor (224 px x scale).")
    p.add_argument("--no-tactile", action="store_true", help="No taxel panel (the window is then 2 cameras wide, as before).")
    p.add_argument("--tactile-width", type=int, default=448,
                   help="Width (px) of the taxel panel, drawn to the right of the cameras; its height is half. 448 = the "
                        "size of the panel in the videos. The window is 2 * 224 * scale + this wide: with --scale 3 that "
                        "is 1792 px; use --scale 2 on a smaller screen.")
    p.add_argument("--tactile-vmax", type=float, default=5.0,
                   help="Force (N) at the top of the colour scale (tactile_vmax of the YAML, so that this panel and the "
                        "videos compare). Display only.")
    p.add_argument("--max-steps", type=int, default=0, help="Episode length (0 = max_steps_per_episode of the server, 150 in the YAML).")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Quit by itself after this time and print the latency summary (0 = no limit).")
    p.add_argument("--continuous", action="store_true",
                   help="Start in AUTO mode: step the simulation at every loop (--hz), also without input, like a "
                        "real-time teleop. Default: step only when you give an input (slow motion). "
                        "The C key switches between the two modes during the session.")
    p.add_argument("--hz", type=float, default=15.0, help="Loop rate of --continuous (the server may be slower).")
    p.add_argument("--hold", action="store_true",
                   help="Do not move the arm (step 1 behaviour): continuous zero moves, to measure the latency.")
    # size of the moves
    p.add_argument("--xy-gain-mm", type=float, default=0.1, help="Peg move (mm) per mouse pixel, in the horizontal plane.")
    p.add_argument("--z-rate-mm", type=float, default=4.0, help="Peg move (mm) per step while W / S is down.")
    p.add_argument("--yaw-rate-deg", type=float, default=3.0,
                   help="Rotation of the hand about the vertical (degrees) per step while Q / E is down (Shift x0.25, Ctrl x2.5). NutThread needs it; 0 disables the keys.")
    p.add_argument("--key-xy-mm", type=float, default=2.0,
                   help="Peg move (mm) per step while an arrow key is down (Shift x0.25, Ctrl x2.5).")
    p.add_argument("--max-xy-mm", type=float, default=10.0, help="Largest horizontal move per step (mm).")
    p.add_argument("--max-z-mm", type=float, default=15.0, help="Largest vertical move per step (mm).")
    p.add_argument("--cmd-gain", type=float, default=1.0,
                   help="Multiplies every joint-target change (as --cmd-gain of collect_waypoint_demos.py).")
    p.add_argument("--max-joint-step", type=float, default=0.05, help="Safety: largest joint-target change (rad).")
    # mouse and arrow directions
    p.add_argument("--view", choices=("wrist", "exterior"), default="wrist",
                   help="Image the mouse and the arrows refer to: the wrist camera (right image, default) or the "
                        "exterior camera (left image).")
    p.add_argument("--flip-x", action="store_true", help="Mouse right moves the peg to the left of the image.")
    p.add_argument("--flip-y", action="store_true", help="Mouse forward moves the peg towards the camera.")
    p.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate the mouse directions in the table plane.")
    args = p.parse_args()

    import pygame
    pygame.init()
    # keys whose short taps (between two slow steps) must not be missed: remembered until the next step
    LATCHED_KEYS = (pygame.K_w, pygame.K_s, pygame.K_q, pygame.K_e, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
    size = 224 * args.scale
    hud_height = 284
    panel_w = 0 if args.no_tactile else max(224, args.tactile_width)      # taxel panel column, right of the cameras
    panel_h = panel_w // 2
    screen = pygame.display.set_mode((max(2 * size + panel_w, 1000), size + hud_height))
    pygame.display.set_caption("FORGE teleop client (step 3)")
    font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 18)
    big_font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 28, bold=True)
    clock = pygame.time.Clock()

    env = RemoteEnv(args.host, args.port)
    env.check_server()
    task_name = env.spec["robot"].get("task")
    if args.out_dir is None:
        args.out_dir = default_out_dir(task_name)
    pygame.display.set_caption(f"FORGE {task_name or 'PegInsert'} -- teleop client (step 3)")
    prompt = env.prompt
    args.max_steps = args.max_steps or int(env.spec["max_steps"])
    geom = env.debug("task_geometry")
    print(f"connected to {env.uri}   task: {task_name or '(not told by the server: PegInsert assumed)'}   prompt: {prompt!r}")
    print(f"hole depth {geom['hole_depth'] * 1000:.0f} mm   success: xy < {geom['success_xy'] * 1000:.1f} mm, "
          f"z < {geom['success_z'] * 1000:.1f} mm")
    env.measure_rtt()
    print(f"network round trip (empty call): {env.mean_ms('rtt'):.1f} ms")

    session = Session(env, args)
    session.prompt = prompt
    session.start(args.seed)
    print(f"recording: demos are written on the cluster in <project>/{args.out_dir}" if args.out_dir
          else "recording OFF (--out-dir '')")
    saved_count = 0                                  # demos saved by this run

    def describe_axes():
        right, up = session.axes
        return (f"{args.view} image: right -> env (x, y) = ({right[0]:+.2f}, {right[1]:+.2f})   "
                f"up -> ({up[0]:+.2f}, {up[1]:+.2f})")
    print(describe_axes())
    print(f"observation keys: {sorted(session.obs.keys())}")
    print("episode", session.seed, "started.  Tab = capture the mouse, R = new episode, Esc = quit.")

    grabbed = False
    auto = bool(args.continuous)              # C toggles: True = the simulation steps at every loop, also without input
    acc = np.zeros(2, dtype=np.int64)         # mouse motion since the previous step
    latched = set()                           # W / S / Space pressed since the previous step (short taps)
    shown = Inputs()                          # last inputs, for the HUD
    t_start = time.perf_counter()
    last_print = t_start
    last_logged = None
    panel_obs, panel_surface = None, None     # the taxel panel is redrawn only when the observation changes
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_z, pygame.K_RETURN, pygame.K_KP_ENTER):
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if not session.success:
                            continue                                  # nothing to save yet
                        if not session.saved and not session.save():
                            continue                                  # the error is printed, Enter tries again
                        saved_count += int(session.saved)
                        new_seed = session.seed + 1
                    else:
                        if session.success and not session.saved:
                            print(f"episode {session.seed}: successful but NOT saved (Enter saves it) -- discarded")
                        new_seed = session.seed if event.key == pygame.K_z else session.seed + 1
                    session.start(new_seed)
                    auto = bool(args.continuous)
                    acc[:] = 0
                    latched.clear()
                    print("episode", session.seed, "started" + (" again (same start state)" if event.key == pygame.K_z else "")
                          + ".   " + describe_axes())
                elif event.key == pygame.K_TAB:
                    grabbed = not grabbed
                    pygame.event.set_grab(grabbed)
                    pygame.mouse.set_visible(not grabbed)
                    pygame.mouse.get_rel()                   # forget the jump caused by the capture
                    acc[:] = 0
                elif event.key == pygame.K_c:
                    auto = not auto
                    print(f"step {session.step_count}: {'AUTO mode (steps counted continuously)' if auto else 'slow motion (steps only on input)'}")
                elif event.key in LATCHED_KEYS:
                    latched.add(event.key)
        keys = pygame.key.get_pressed()
        dx, dy = pygame.mouse.get_rel()
        if grabbed:
            acc += (dx, dy)
        else:
            acc[:] = 0
            latched.clear()

        inp = Inputs(dx=int(acc[0]), dy=int(acc[1]),
                     up=bool(keys[pygame.K_w] or pygame.K_w in latched),
                     down=bool(keys[pygame.K_s] or pygame.K_s in latched),
                     fine=bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
                     fast=bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]),
                     wait=pygame.K_SPACE in latched,
                     left=bool(keys[pygame.K_LEFT] or pygame.K_LEFT in latched),
                     right=bool(keys[pygame.K_RIGHT] or pygame.K_RIGHT in latched),
                     forward=bool(keys[pygame.K_UP] or pygame.K_UP in latched),
                     back=bool(keys[pygame.K_DOWN] or pygame.K_DOWN in latched),
                     yaw_left=bool(args.yaw_rate_deg and (keys[pygame.K_q] or pygame.K_q in latched)),
                     yaw_right=bool(args.yaw_rate_deg and (keys[pygame.K_e] or pygame.K_e in latched)))
        stepping = False
        if not session.over:
            if args.hold or auto:
                stepping = grabbed or args.hold
            else:
                stepping = grabbed and inp.any()
        if stepping:
            if args.hold:
                inp = Inputs()
            delta = build_delta(inp, session.axes, args.xy_gain_mm, args.z_rate_mm, args.max_xy_mm, args.max_z_mm,
                                args.key_xy_mm)
            session.step(delta, build_yaw(inp, args.yaw_rate_deg))
            shown = inp
            acc[:] = 0
            latched.clear()
            if session.success:
                print(f"episode {session.seed}: SUCCESS after {session.step_count} steps")
            elif session.timed_out:
                print(f"episode {session.seed}: over after {session.step_count} steps (no success)")

        # -- draw --
        screen.fill(BACKGROUND)
        screen.blit(to_surface(pygame, session.obs["observation/exterior_image_1_left"], args.scale), (0, 0))
        screen.blit(to_surface(pygame, session.obs["observation/wrist_image_left"], args.scale), (size, 0))
        if panel_w:
            if panel_obs is not session.obs:
                effort = session.obs.get("extra/gripper_effort")
                panel_surface = tactile_panel_surface(
                    pygame, tactile_from_obs(session.obs), panel_w, panel_h, args.tactile_vmax,
                    hand_force=session.force_n(),
                    gripper_effort=None if effort is None else float(np.asarray(effort, dtype=np.float64).reshape(-1)[0]))
                panel_obs = session.obs
            screen.blit(panel_surface, (2 * size, 0))
        offset_mm = session.offset_mm()
        xy = float(np.hypot(offset_mm[0], offset_mm[1]))
        xy_colour = GOOD if xy < geom["success_xy"] * 1000 else TEXT
        y = size + 6
        x = 10
        for text, colour in ((f"xy {xy:6.2f} mm", xy_colour), (f"z {offset_mm[2]:5.1f} mm", TEXT),
                             (f"x {offset_mm[0]:+6.2f}  y {offset_mm[1]:+6.2f}", DIM)):
            rendered = big_font.render(text, True, colour)
            screen.blit(rendered, (x, y))
            x += rendered.get_width() + 30
        y += 42
        if session.success and session.saved:
            status, status_colour = "SAVED", GOOD
        elif session.success:
            status, status_colour = ("SUCCESS -- Enter = SAVE + next seed, Z = redo, R = skip" if args.out_dir
                                     else "SUCCESS (recording off) -- R = next seed"), GOOD
        elif session.over:
            status, status_colour = "episode over -- Z = same seed again, R = next seed", WARN
        elif not grabbed and not args.hold:
            status, status_colour = "mouse FREE -- press Tab to take control", WARN
        else:
            status, status_colour = ("AUTO: steps counted continuously (C = stop)" if auto or args.hold else "your move (C = auto steps)"), (WARN if auto else TEXT)
        over_soft = session.step_count > args.soft_limit_steps and not session.over
        y = draw_text(screen, font, 10, y,
                      f"seed {session.seed}   step {session.step_count}/{args.max_steps}   {status}",
                      WARN if over_soft and status_colour == TEXT else status_colour)
        d = session.last_delta * 1000.0 + 0.0            # "+ 0.0": no "-0.00" for a negative zero
        force = session.force_n()
        y = draw_text(screen, font, 10, y,
                      f"last move (mm)  x {d[0]:+6.2f}  y {d[1]:+6.2f}  z {d[2]:+6.2f}   hand yaw {np.rad2deg(session.yaw):+6.1f} deg"
                      f"{'   JOINT CLIP' if session.last_clipped else ''}"
                      f"      contact force {force:5.2f} N", WARN if force > 2.0 else TEXT)
        total = env.mean_ms("teleop_step")
        y = draw_text(screen, font, 10, y,
                      f"saved this run: {saved_count}   (success hold {session.success_run}/{HOLD_AFTER_SUCCESS})"
                      + ("   -- long demo, try to be quicker" if over_soft else ""), WARN if over_soft else DIM)
        y = draw_text(screen, font, 10, y,
                      f"ms mean/p95:  teleop_step {total:.0f}/{env.p95_ms('teleop_step'):.0f}   "
                      f"empty call {env.mean_ms('rtt'):.0f}   -> {1000.0 / total if total == total else float('nan'):.1f} steps/s max",
                      DIM)
        mode = "FINE" if shown.fine else ("FAST" if shown.fast else "normal")
        down_keys = [name for name, key in (("W", pygame.K_w), ("S", pygame.K_s), ("Q", pygame.K_q), ("E", pygame.K_e), ("Space", pygame.K_SPACE),
                                            ("Left", pygame.K_LEFT), ("Right", pygame.K_RIGHT), ("Up", pygame.K_UP),
                                            ("Down", pygame.K_DOWN), ("Shift", pygame.K_LSHIFT),
                                            ("Ctrl", pygame.K_LCTRL)) if keys[key]]
        y = draw_text(screen, font, 10, y,
                      f"inputs: mouse pending ({acc[0]:+4d},{acc[1]:+4d}) px   keys: {' '.join(down_keys) or '-'}   "
                      f"mode: {mode}   mouse {'CAPTURED' if grabbed else 'free'}", DIM)
        y = draw_text(screen, font, 10, y,
                      f"{args.view} image: right -> env ({session.axes[0][0]:+.2f},{session.axes[0][1]:+.2f})  up -> "
                      f"({session.axes[1][0]:+.2f},{session.axes[1][1]:+.2f})   mouse {args.xy_gain_mm:g} mm/px   "
                      f"arrows {args.key_xy_mm:g} mm/step   "
                      f"W/S {args.z_rate_mm:g} mm/step", DIM)
        y = draw_text(screen, font, 10, y,
                      "Tab capture  mouse/arrows = xy (wrist)  W/S up/down  Q/E turn hand  Shift fine  Ctrl fast  Space wait", DIM)
        draw_text(screen, font, 10, y,
                  "C auto steps  Enter save (after SUCCESS)  Z same seed again  R next seed  Esc quit", DIM)
        pygame.display.flip()

        # -- pace the loop --
        if auto or args.hold:
            clock.tick(args.hz if args.hz > 0 else 0)
        else:
            clock.tick(60)                      # idle: only redraw, the simulation is not stepped
        now = time.perf_counter()
        if now - last_print > 5.0 and env.timings.get("teleop_step"):
            last_print = now
            if (session.seed, session.step_count) != last_logged:      # nothing new while idle / after the end
                last_logged = (session.seed, session.step_count)
                print(f"  teleop_step {env.mean_ms('teleop_step'):.0f} ms (p95 {env.p95_ms('teleop_step'):.0f})   "
                      f"step {session.step_count}")
        if args.max_seconds and now - t_start > args.max_seconds:
            running = False

    print("\nround trips (ms)        mean     p95")
    for name in ("rtt", "teleop_step"):
        if env.timings.get(name):
            label = "empty call (network)" if name == "rtt" else "teleop_step"
            print(f"  {label:<20s} {env.mean_ms(name):7.1f} {env.p95_ms(name):7.1f}")
    if env.timings.get("teleop_step"):
        total = env.mean_ms("teleop_step")
        print(f"  -> {1000.0 / total:.1f} steps/s at most; a {args.max_steps}-step episode needs at least "
              f"{args.max_steps * total / 1000.0:.0f} s of pure simulation")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
