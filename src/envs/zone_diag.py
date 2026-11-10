"""Diagnostic printed by the server when the tactile contact sensors cannot be created (see forge_backend.build_env).

Isaac Lab's ContactSensor raises "Failed to initialize contact reporter for specified bodies" when PhysX's rigid-body view of
the zone bodies has not exactly `bodies per env x num_envs` members. The exception does not say how many it found, nor why.
This prints, from the stage and from PhysX itself (both still alive after the failure), what is needed to tell the causes apart:
  * what the zone prim of env_0 is in USD (type, rigid body, enabled, kinematic, contact report, instance proxy);
  * how many rigid bodies of each kind USD declares under TactileZones;
  * how many bodies PhysX finds for several path patterns, from the exact one Isaac Lab used down to a single body, plus a
    control body that certainly exists (panda_link0).
Every probe is wrapped on its own: this function must never hide the original error.

Also here (same purpose, temporary):
  * DiagContactSensor: Isaac Lab's ContactSensor that prints, at the very moment its initialisation fails, the numbers its own
    check compares (number of environments it counted, bodies it found, what PhysX's view has at that instant).
  * print_sensor_buffers(env): the size of every sensor's buffers, to find the one whose reset indexes out of range.
"""
from __future__ import annotations

from isaaclab.sensors import ContactSensor

ZONE_ROOT = "/World/envs/env_0/Robot/TactileZones"


def _try(label, fn):
    try:
        print(f"[zone_diag] {label}: {fn()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[zone_diag] {label}: probe failed ({type(e).__name__}: {e})", flush=True)


def print_zone_diagnostics(sides=("left", "right"), rows=7, cols=4):
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()
    print("[zone_diag] ---- tactile zone diagnostic (the sensor creation failed) ----", flush=True)

    def describe(path):
        p = stage.GetPrimAtPath(path)
        if not p.IsValid():
            return "NO SUCH PRIM"
        attr = lambda n: (p.GetAttribute(n).Get() if p.HasAttribute(n) else None)  # noqa: E731
        return (f"type={p.GetTypeName() or '-'} rigid_body={p.HasAPI(UsdPhysics.RigidBodyAPI)} "
                f"rigidBodyEnabled={attr('physics:rigidBodyEnabled')} kinematic={attr('physics:kinematicEnabled')} "
                f"collision={p.HasAPI(UsdPhysics.CollisionAPI)} contact_report={p.HasAPI(PhysxSchema.PhysxContactReportAPI)} "
                f"instanceable={p.IsInstanceable()} instance_proxy={p.IsInstanceProxy()} "
                f"children={[c.GetName() for c in p.GetChildren()][:6]}")

    _try("USD zone_r0_c0 (left)", lambda: describe(f"{ZONE_ROOT}/left/zone_r0_c0"))
    _try("USD TactileZones prim", lambda: describe(ZONE_ROOT))

    def count_bodies():
        from pxr import Usd
        out = {}
        for side in sides:
            root = stage.GetPrimAtPath(f"{ZONE_ROOT}/{side}")
            prims = [q for q in Usd.PrimRange(root)] if root.IsValid() else []
            out[side] = {"rigid_bodies": sum(q.HasAPI(UsdPhysics.RigidBodyAPI) for q in prims),
                         "contact_report": sum(q.HasAPI(PhysxSchema.PhysxContactReportAPI) for q in prims),
                         "fixed_joints": sum(q.IsA(UsdPhysics.FixedJoint) for q in prims)}
        return out
    _try("USD counts under TactileZones/<side>", count_bodies)

    sim_view = SimulationManager.get_physics_sim_view()
    names = [f"zone_r{r}_c{c}" for r in range(rows) for c in range(cols)]
    alternation = "(" + "|".join(names) + ")"
    base = "/World/envs/env_*/Robot/TactileZones/left"
    probes = [
        ("PhysX view, exact pattern used by Isaac Lab", f"{base}/{alternation}"),
        ("PhysX view, wildcard zone_r*_c*", f"{base}/zone_r*_c*"),
        ("PhysX view, env_0 only, wildcard", "/World/envs/env_0/Robot/TactileZones/left/zone_r*_c*"),
        ("PhysX view, env_0 only, ONE zone", "/World/envs/env_0/Robot/TactileZones/left/zone_r0_c0"),
        ("PhysX view, alternation of 2 zones", f"{base}/(zone_r0_c0|zone_r0_c1)"),
        ("PhysX view, both sides, wildcard", "/World/envs/env_*/Robot/TactileZones/*/zone_r*_c*"),
        ("PhysX view, CONTROL panda_link0", "/World/envs/env_*/Robot/robot/panda_link0"),
    ]
    for label, pattern in probes:
        _try(label + f"  [{pattern if len(pattern) < 90 else pattern[:60] + '...'}]",
             lambda pattern=pattern: f"count={sim_view.create_rigid_body_view(pattern).count}")
    print("[zone_diag] expected count with the YAML num_envs: 28 x num_envs for the first patterns, 1 x num_envs for ONE zone and "
          "control, 2 x num_envs for the 2-zone alternation, 56 x num_envs for both sides.", flush=True)


class DiagContactSensor(ContactSensor):
    """ContactSensor + a printout of the quantities compared by its failing check, taken INSIDE the failing call."""

    def _initialize_impl(self):
        try:
            super()._initialize_impl()
        except RuntimeError:
            self._print_failure_state()
            raise

    def _print_failure_state(self):
        import isaaclab.sim as sim_utils
        from pxr import PhysxSchema
        from isaacsim.core.simulation_manager import SimulationManager
        tag = f"[zone_diag] (inside the failing sensor init, {self.cfg.prim_path.rsplit('/', 1)[0].rsplit('/', 1)[-1]})"
        _try(tag + " envs counted by the sensor (_num_envs)", lambda: getattr(self, "_num_envs", "unset"))
        _try(tag + " parent prims it matched", lambda: [p.GetPath().pathString for p in self._parent_prims])
        _try(tag + " count of ITS body view (_body_physx_view)",
             lambda: self._body_physx_view.count if getattr(self, "_body_physx_view", None) is not None else "unset")

        def names():
            template = self._parent_prims[0].GetPath().pathString
            prims = sim_utils.find_matching_prims(template + "/" + self.cfg.prim_path.rsplit("/", 1)[-1])
            with_api = [p.GetName() for p in prims if p.HasAPI(PhysxSchema.PhysxContactReportAPI)]
            return f"matched={len(prims)} with contact-report API={len(with_api)} distinct={len(set(with_api))}"
        _try(tag + " bodies found by the sensor", names)

        def now():
            view = SimulationManager.get_physics_sim_view()
            return view.create_rigid_body_view("/World/envs/env_*/Robot/TactileZones/left/zone_r*_c*").count
        _try(tag + " PhysX wildcard view count AT THIS INSTANT", now)


def print_sensor_buffers(env):
    """One line per sensor of the scene (type, prim path, instances, size of its timestamp buffer), then a plain CUDA indexing
    check: a sensor whose reset raises a device-side assert at `self._timestamp[env_ids] = 0.0` has a buffer smaller than num_envs."""
    import torch
    print(f"[zone_diag] ---- sensors of the scene (num_envs={env.num_envs}) ----", flush=True)
    for name, s in env.scene.sensors.items():
        _try(f"sensor {name}", lambda s=s: (
            f"{type(s).__name__} prim_path={s.cfg.prim_path} instances={s.num_instances} "
            f"timestamp_shape={tuple(s._timestamp.shape)} device={s._timestamp.device}"))

    def plain_index():
        t = torch.zeros(env.num_envs, device=env.device)
        ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        t[ids] = 0.0
        torch.cuda.synchronize()
        return "ok"
    _try("plain index_put on a (num_envs,) CUDA tensor", plain_index)
