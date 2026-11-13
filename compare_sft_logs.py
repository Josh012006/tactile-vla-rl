"""Side-by-side training curves of our SFT (runs/.../metrics.jsonl) and expo-ft's (openpi log `Step N: grad_norm=..., loss=..., param_norm=...`), reading files only.

    python compare_sft_logs.py --ours runs/sft/peg_insert/peg_insert_pi05_lora_teleop/metrics.jsonl --theirs ~/projects/expo-ft/logs/forge_sft_11436287.out

openpi logs every 100 steps the MEAN of the 100 steps since the last line (step 0 alone on its own line); ours logs the mean of the 50 steps since the last line. Our two windows that
end in (N - 50, N] are averaged to compare with theirs; the first window of ours also holds the very first step (loss of the untrained model), theirs does not (its step 0 is a separate line).
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--theirs", required=True)
args = ap.parse_args()

ours = {}
for line in open(args.ours):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("event") == "train":
        ours[r["step"]] = r
pat = re.compile(r"Step (\d+): (.*)")
theirs = {}
for line in open(args.theirs, errors="replace"):
    m = pat.search(line)
    if m:
        theirs[int(m.group(1))] = {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", m.group(2))}
print(f"{'step':>5} | {'loss ours':>10} {'theirs':>9} {'ours/theirs':>11} | {'gnorm ours':>10} {'theirs':>8} {'ratio':>6} | {'pnorm ours':>11} {'theirs':>11} {'diff':>7}")
if 0 in theirs:
    print("theirs step 0:", theirs[0])
for step in sorted(theirs):
    if step == 0 or step not in ours or (step - 50) not in ours:
        continue
    lo = (ours[step]["loss"] + ours[step - 50]["loss"]) / 2
    go = (ours[step]["grad_norm"] + ours[step - 50]["grad_norm"]) / 2
    t = theirs[step]
    print(f"{step:5d} | {lo:10.5f} {t['loss']:9.5f} {lo / t['loss']:11.2f} | {go:10.4f} {t['grad_norm']:8.4f} {go / t['grad_norm']:6.2f} | "
          f"{ours[step].get('param_norm', float('nan')):11.4f} {t.get('param_norm', float('nan')):11.4f} {ours[step].get('param_norm', float('nan')) - t.get('param_norm', float('nan')):+7.4f}")
