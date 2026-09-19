#!/usr/bin/env python3
"""oracle_aggregate.py - summarise oracle_sweep.py outputs (Q2 of AUTOPILOT.md).

Reads any number of oracle_*.json state dumps and reports, over the WHOLE sample:
  * disagreement rate with its sample size (never a single example)
  * which behaviour wins, and how often a disagreeing state has MANY winners at once
    (a state where 4-6 of 6 alternatives all "win" is the signature of a harness artefact,
     not a lesson about behaviour - that is what this script is here to catch)
  * breakdown by state features (lockout, fruit visible, predator visible)
  * whether the controller's focal agent was alive at the end of the branch

Usage: ./oracle_aggregate.py oracle_x1.json oracle_serv1.json ...
       ./oracle_aggregate.py --dir experiments/wb_evidence/oracle_serv
"""
import argparse
import collections
import glob
import json
import os
import statistics as st


def load(paths):
    rows = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        if not isinstance(d, list) or not d or not isinstance(d[0], dict) or "beats" not in d[0]:
            print(f"  skip {os.path.basename(p)}: not a state dump")
            continue
        for s in d:
            s["_file"] = os.path.basename(p)
            rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", help="directory of oracle_*.json")
    a = ap.parse_args()
    paths = list(a.files)
    if a.dir:
        paths += sorted(glob.glob(os.path.join(a.dir, "oracle_*.json")))
    rows = load(paths)
    if not rows:
        print("no states")
        return
    n = len(rows)
    dis = [s for s in rows if s["beats"]]
    print(f"states={n}  files={len(set(s['_file'] for s in rows))}  "
          f"disagreements={len(dis)}  rate={100*len(dis)/n:.0f}%")

    wcount = collections.Counter()
    nwin = collections.Counter()
    for s in dis:
        nwin[len(s["winners"])] += 1
        for w in s["winners"]:
            wcount[w] += 1
    print("winners per disagreeing state:", dict(sorted(nwin.items())))
    print("win counts:", wcount.most_common())
    if dis:
        multi = sum(v for k, v in nwin.items() if k >= 3)
        print(f"NOTE: {multi}/{len(dis)} disagreeing states have >=3 winners at once "
              f"({100*multi/len(dis):.0f}%) - check the harness before believing the winner")

    def bucket(s):
        f = s["features"]
        return f"lockout={int(f['in_lockout'])} fruit={int(f['fruit_visible']>0)} pred={int(f['pred_visible']>0)}"
    bg = collections.defaultdict(lambda: [0, 0])
    for s in rows:
        bg[bucket(s)][1] += 1
        if s["beats"]:
            bg[bucket(s)][0] += 1
    print(f"\n{'bucket':>34} {'states':>7} {'beats':>6} {'rate':>6}")
    for k, (b, tot) in sorted(bg.items(), key=lambda x: -x[1][1]):
        print(f"{k:>34} {tot:>7} {b:>6} {100*b/tot:>5.0f}%")

    alive = [s for s in rows if s["controller"]["alive"] and s["controller"]["energy"] > 0]
    da = [s for s in alive if s["beats"]]
    print(f"\ncontroller focal agent alive with energy>0 at end of branch: {len(alive)}/{n} "
          f"({100*len(alive)/n:.0f}%)  -> disagreement there {len(da)}/{max(len(alive),1)} "
          f"({100*len(da)/max(len(alive),1):.0f}%)")
    gains = []
    for s in alive:
        c = s["controller"]["energy"]
        gains.append(max(v["energy"] for v in s["alternatives"].values()) - c)
    if gains:
        gains.sort()
        print(f"best-alternative minus controller energy (alive states): "
              f"median {st.median(gains):+.1f}  p10 {gains[int(.1*len(gains))]:+.1f}  "
              f"p90 {gains[int(.9*len(gains))]:+.1f}  max {gains[-1]:+.1f}")


if __name__ == "__main__":
    main()
