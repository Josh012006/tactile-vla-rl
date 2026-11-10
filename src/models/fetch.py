"""Download everything the VLA models need, ONCE, on a node that has internet (the login node), so that the jobs run offline.

    source .venv/bin/activate && python -m src.models.fetch            # every configs/model/*.yaml
    python -m src.models.fetch --model pi0_fast

For each model YAML: the checkpoint (params + assets with the norm stats) into openpi's cache (OPENPI_DATA_HOME, default
~/.cache/openpi) and, for the tokenizers openpi loads at runtime, the PaliGemma sentencepiece model (gs://big_vision) and, for
pi0-FAST, the FAST tokenizer from the Hugging Face Hub (cache HF_HOME). It prints the Hub revision of the FAST tokenizer: it is the
version the experiments ran with (the submodule does not pin it), write it in configs/model/pi0_fast.yaml.
Jobs then set HF_HUB_OFFLINE=1: a file that was not fetched fails loudly instead of being downloaded silently.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FAST_TOKENIZER = "physical-intelligence/fast"
# openpi-assets is a PUBLIC bucket. Without gsutil, openpi falls back to gcsfs, which looks for Google credentials (a long hang on a
# cluster node without any) unless it is told to read anonymously.
GCS_ANON = {"token": "anon"}


def size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2 ** 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="configs/model/<model>.yaml (default: all)")
    a = ap.parse_args(argv)
    names = a.model or sorted(p.stem for p in (ROOT / "configs" / "model").glob("*.yaml"))

    import openpi.shared.download as download
    from openpi.models import tokenizer as _tok

    print(f"OPENPI_DATA_HOME={os.environ.get('OPENPI_DATA_HOME', '~/.cache/openpi (default)')}  HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")
    fast = False
    for name in names:
        cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{name}.yaml").read_text())
        print(f"[{name}] checkpoint {cfg['checkpoint']} ...", flush=True)
        path = download.maybe_download(cfg["checkpoint"], gs=GCS_ANON)
        for sub in ("params", "assets"):
            assert (Path(path) / sub).exists(), f"{path}/{sub} is missing: the checkpoint is not what openpi expects"
        print(f"[{name}] ok: {path}  ({size_gib(path):.1f} GiB)")
        if cfg.get("family") == "pi0_fast":
            fast = True
            _tok.FASTTokenizer(max_len=int(cfg.get("max_token_len", 180)))      # downloads the PaliGemma model and the FAST tokenizer
        else:
            _tok.PaligemmaTokenizer()
    if fast:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(FAST_TOKENIZER))
        print(f"FAST tokenizer {FAST_TOKENIZER}: revision {snap.name}  ({snap})")
        print("  -> write it in configs/model/pi0_fast.yaml (fast_tokenizer_revision) so the run records the version it used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
