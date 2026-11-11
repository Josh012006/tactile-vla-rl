"""Task progress of a FORGE peg-insertion rollout, from the env's own observation keys. numpy only.

`extra/peg_socket_offset` (held peg base minus its target pose in the socket, metres, env frame: xy = norm(offset[:2]) is FORGE's `xy_dist`,
z its `z_disp`; FORGE's insertion test needs xy < 2.5 mm and z below a height threshold) and `extra/ft_force` (N). Nothing is added to the
env protocol.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.envs import protocol as P

XY_INSERTED_MM = 2.5        # FORGE's xy tolerance for "inserted" (forge_env.py: inserted())


def progress_sample(obs: Dict[str, np.ndarray], env: int) -> dict:
    """Peg-to-socket distances and contact force of environment `env` in the observation of a step (mm, N)."""
    off = np.asarray(obs[P.PEG_SOCKET_OFFSET][env], dtype=np.float64)
    force = np.asarray(obs[P.FT_FORCE][env], dtype=np.float64)
    return {"xy_mm": float(1e3 * np.linalg.norm(off[:2])), "z_mm": float(1e3 * off[2]), "force_n": float(np.linalg.norm(force))}


def progress_summary(series: List[dict], success: bool) -> dict:
    """series: one dict per step ({"t", "xy_mm", "z_mm", "force_n"}), the first one is the state after the reset."""
    if not series:
        return {"samples": 0, "success": bool(success)}
    xy, z, f = (np.array([s[k] for s in series]) for k in ("xy_mm", "z_mm", "force_n"))
    i = int(xy.argmin())
    return {"samples": len(series), "xy_mm_start": float(xy[0]), "xy_mm_end": float(xy[-1]), "xy_mm_min": float(xy[i]),
            "step_of_xy_min": int(series[i]["t"]), "z_mm_start": float(z[0]), "z_mm_end": float(z[-1]), "z_mm_min": float(z.min()),
            "force_n_max": float(f.max()), "steps_xy_within_tolerance": int((xy < XY_INSERTED_MM).sum()), "success": bool(success)}
