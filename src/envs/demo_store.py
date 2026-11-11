"""Writing a demonstration sent by a client, on the machine of the env server (numpy only: no Isaac Sim, tested on the CPU).

Why here: the teleoperation client (scripts/demos/teleop_client.py) runs on the operator's PC and only holds the episode in memory. When
the episode is over it builds the .npz and sends its bytes in ONE message (debug op `save_teleop_demo`); the server, which sees the
cluster's disk, writes the file. A half-written demo never exists: the bytes are written to a temporary file and renamed.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path
from typing import Union

import numpy as np

# Arrays every demo must have (the keys of scripts/demos/collect_waypoint_demos.py, which the conversion for the SFT reads).
REQUIRED_ARRAYS = ("exterior_image", "wrist_image", "joint_position", "gripper_position", "ft_force", "peg_socket_offset",
                   "actions", "prompt", "seed")


def save_teleop_demo(project_root: Union[str, Path], request: dict) -> dict:
    """Write one teleoperated episode as <out_dir>/ep_NNNN_seed<seed>.npz (or fail_seed<seed>_<time>.npz when it did not succeed) and
    append a line to <out_dir>/index.jsonl.

    request: out_dir (relative to `project_root`, must stay inside it), seed, npz (bytes), info (dict, one line of index.jsonl), success.
    NNNN continues after the highest number already in the folder. out_dir is checked WITHOUT following symlinks: `demos` is a link to a
    scratch disk and that must keep working. Returns {"file", "path", "count"} or {"error": ...} (nothing is written then)."""
    root = Path(project_root)
    out = Path(os.path.normpath(root / str(request["out_dir"])))
    if root != out and root not in out.parents:
        return {"error": f"out_dir must be inside the project ({root}), got {request['out_dir']!r}"}
    seed = int(request["seed"])
    data = bytes(request["npz"])
    with np.load(io.BytesIO(data)) as z:                      # refuse anything that is not a complete demo
        missing = sorted(set(REQUIRED_ARRAYS) - set(z.files))
        if missing:
            return {"error": f"the .npz lacks {missing}"}
    out.mkdir(parents=True, exist_ok=True)
    if bool(request["success"]):
        numbers = [int(m.group(1)) for f in out.glob("ep_*.npz") if (m := re.match(r"ep_(\d+)_", f.name))]
        fname = f"ep_{(max(numbers) + 1) if numbers else 0:04d}_seed{seed}.npz"
    else:
        fname = f"fail_seed{seed}_{time.strftime('%H%M%S')}.npz"
    tmp = out / f".tmp_{os.getpid()}.npz"                     # written fully, then renamed: no half-written demo
    tmp.write_bytes(data)
    os.replace(tmp, out / fname)
    with open(out / "index.jsonl", "a") as index:
        index.write(json.dumps({**request["info"], "file": fname}) + "\n")
    return {"file": fname, "path": str(out / fname), "count": len(list(out.glob("ep_*.npz")))}
