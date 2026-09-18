#!/usr/bin/env python3
"""paired_stats.py — turn a batch_par / evolve_meta log into a defensible paired verdict.

Why this exists: three times tonight a headline number was wrong because it was read off an
unpaired mean (E3's +9.5%, V2's +12.2%, and the phase-test baseline). A paired comparison needs
per-seed deltas, a bootstrap CI on the paired mean, and an explicit win/loss count — and the
decision rule has to be checked mechanically, not by eye.

Usage:
    .venv/bin/python tools/ops/paired_stats.py <reference_tag> <logfile> [<logfile> ...]

Log format expected (one line per arm, as batch_par.py prints it):
    <tag>  med_ticks=  7124 mean_ticks=  7233 med_score= 712.9 min= ... ticks=[7170, 8417, ...]
Any other lines are ignored, so evolve_meta logs can be piped in too as long as the per-seed
`ticks=[...]` list is present.
"""
import re
import subprocess
import sys

import numpy as np

LINE = re.compile(r'^(?P<tag>\S+)\s+.*?ticks=\[(?P<ticks>[0-9,\s]*)\]')
DEFAULT_RULE = ">=60% paired wins AND mean gain above the noise band"


def load(paths, via_ssh=None):
    """Reads local paths, or remote ones (a path that does not exist locally is fetched over SSH)."""
    import os
    rows = {}
    for p in paths:
        if via_ssh:
            text = subprocess.run(["ssh", "-i", "/Users/zaitzev/.ssh/vps_hermes", via_ssh, f"cat {p}"],
                                  capture_output=True, text=True).stdout
        elif os.path.exists(p):
            text = open(p).read()
        else:
            text = subprocess.run(["ssh", "-i", "/Users/zaitzev/.ssh/vps_hermes",
                                   os.environ.get("NAC_HOST", "root@94.237.34.245"), f"cat {p}"],
                                  capture_output=True, text=True).stdout
        for line in text.splitlines():
            m = LINE.match(line.strip())
            if m and m.group("ticks").strip():
                rows[m.group("tag")] = np.array([int(x) for x in m.group("ticks").split(",")], dtype=float)
    return rows


def paired(ref, arm, n_boot=20000, seed=0):
    n = min(len(ref), len(arm))
    d = arm[:n] - ref[:n]
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    return dict(n=n, mean=d.mean(), pct=100 * d.mean() / ref[:n].mean(),
                ci=(lo, hi), median=float(np.median(d)), wins=wins, losses=losses,
                ties=int((d == 0).sum()), win_frac=wins / max(1, wins + losses))


def verdict(row):
    """Mechanical check of the pre-registered rule; never eyeball the number."""
    ok_wins = row["win_frac"] >= 0.60
    ok_gain = row["mean"] > 0 and row["ci"][0] > 0
    if ok_wins and ok_gain:
        return "ADOPT"
    if row["mean"] < 0 and row["ci"][1] < 0:
        return "REFUTED (significant harm)"
    if row["mean"] > 0 and row["ci"][0] > 0:
        return "gain is real but below the win-rate bar"
    return "NO EVIDENCE (CI spans zero) - do not deploy"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ref_tag, paths = sys.argv[1], sys.argv[2:]
    rows = load(paths)
    if ref_tag not in rows:
        print(f"reference tag {ref_tag!r} not found. tags present: {list(rows)}")
        sys.exit(1)
    ref = rows[ref_tag]
    print(f"reference {ref_tag}: n={len(ref)} mean={ref.mean():.0f} median={np.median(ref):.0f} "
          f"min={ref.min():.0f} max={ref.max():.0f}")
    print(f"rule: {DEFAULT_RULE}\n")
    for tag, arm in sorted(rows.items()):
        if tag == ref_tag:
            continue
        r = paired(ref, arm)
        print(f"{tag:24s} n={r['n']:3d}  paired mean {r['mean']:+8.1f} ({r['pct']:+5.1f}%)  "
              f"95% CI [{r['ci'][0]:+.0f}, {r['ci'][1]:+.0f}]  median {r['median']:+7.1f}  "
              f"wins {r['wins']:3d} / losses {r['losses']:3d} / ties {r['ties']:3d}  "
              f"({100*r['win_frac']:.0f}% wins)")
        print(f"{'':24s} -> {verdict(r)}")


if __name__ == "__main__":
    main()
