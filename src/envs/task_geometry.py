"""What FORGE's own success test measures, for each of its three tasks (numpy-free, no Isaac Sim: testable anywhere).

Isaac Lab's `factory_env._get_curr_successes` calls an environment successful when the held object is centred on its target (xy distance below
SUCCESS_XY) AND has gone down far enough (z displacement below a height threshold). The height threshold is, per task:

    peg_insert, gear_mesh   fixed_asset_cfg.height * task.success_threshold          (the depth of the hole / of the shaft the gear goes on)
    nut_thread              fixed_asset_cfg.thread_pitch * task.success_threshold    (the nut has only to go down by about one pitch of the thread)

The server's `task_geometry` debug op reports the numbers (the demo tools and the teleop client print them); keep this the one place that knows
which task uses which.
"""
from typing import Any, Dict

SUCCESS_XY = 0.0025                      # m, FORGE's own constant (`xy_dist < 0.0025`)
TASKS = ("peg_insert", "gear_mesh", "nut_thread")


def success_z(task_name: str, fixed_asset_cfg: Any, success_threshold: float) -> float:
    """The height (m, above the fully inserted / threaded pose) below which the held object counts as inserted."""
    if task_name in ("peg_insert", "gear_mesh"):
        return float(fixed_asset_cfg.height) * float(success_threshold)
    if task_name == "nut_thread":
        return float(fixed_asset_cfg.thread_pitch) * float(success_threshold)
    raise NotImplementedError(f"FORGE task {task_name!r}: known tasks are {TASKS}")


def task_geometry(task_name: str, fixed_asset_cfg: Any, success_threshold: float) -> Dict[str, float]:
    """{"hole_depth": height of the fixed asset (m), "success_xy", "success_z"}: the keys the demo tools and the teleop client read (the name
    `hole_depth` is the one PegInsert gave it; for the other tasks it is the height of the fixed asset all the same)."""
    return {"hole_depth": float(fixed_asset_cfg.height), "success_xy": SUCCESS_XY, "success_z": success_z(task_name, fixed_asset_cfg, success_threshold)}
