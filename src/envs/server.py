"""Isaac Lab environment server (run inside the Isaac Lab virtualenv ONLY).

    export CUBLAS_WORKSPACE_CONFIG=:4096:8        # required for reproducible resets, set it BEFORE the process starts
    python -m src.envs.server --config configs/task/peg_insert_forge.yaml
    python -m src.envs.server --config ... --set num_envs=16 --set robot=robotiq_2f85_tsf     # any YAML field, one run

The process owns the simulation: `num_envs` independent environments in ONE scene (protocol.py), served on `server_port`. They
start their episodes together, with FORGE's own reset (a round), and each runs its own episode; the next round starts when the
last one has ended. The learner is the client (src/envs/client.py). The task is fixed for the life of the process: loading
another scene would cost the ~2 minute Isaac Sim start-up again.

Why everything is inside main(): AppLauncher must exist before any isaaclab_tasks / omni import, and importing this
module (e.g. for `--help` or by a test) must not start Kit.
"""
import argparse
import logging
import os
import signal
import sys
import traceback


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="task YAML (configs/task/*.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a YAML field for this run (repeatable); the value is read as YAML")
    AppLauncher.add_app_launcher_args(parser)
    return parser, parser.parse_args()


def load_config(parser, args) -> dict:
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        cfg[key.strip()] = yaml.safe_load(value)
    return cfg


def main():
    parser, args = parse_args()
    cfg = load_config(parser, args)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        print("[env_server] WARNING: CUBLAS_WORKSPACE_CONFIG is not set (export CUBLAS_WORKSPACE_CONFIG=:4096:8 before "
              "starting): reset(seed) will raise or will not be reproducible.", flush=True)

    from isaaclab.app import AppLauncher
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app

    # Everything that imports Isaac Sim comes after AppLauncher.
    from .forge_backend import build_backend
    from .ws_server import WsEnvServer

    logging.basicConfig(level=logging.INFO)
    try:
        backend, env = build_backend(cfg, device=args.device)
        server = WsEnvServer(backend, host=cfg.get("server_host", "0.0.0.0"), port=int(cfg.get("server_port", 8102)))
    except Exception:  # noqa: BLE001
        # After an exception Kit keeps the process alive (its threads never finish): the server would hang forever with
        # the traceback in its log and whoever started it (a test, a job) would wait for nothing. Print it and exit NOW.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    # AppLauncher's own SIGINT handler tries a Kit-internal shutdown that never returns (Ctrl+C did nothing, the GPU stayed
    # allocated), and Kit re-installs its handlers while the scene is built: so ours is registered LAST, here.
    def shutdown(signum, frame):
        print(f"[env_server] received {signal.Signals(signum).name}: shutting down.", flush=True)
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            logging.exception("error during shutdown (exiting anyway)")
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"[env_server] ready: {backend.manager.n} environments, port {server.port or cfg.get('server_port', 8102)}", flush=True)
    server.serve_forever()      # returns only through shutdown() -> os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
