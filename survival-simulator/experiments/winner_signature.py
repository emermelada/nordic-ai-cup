#!/usr/bin/env python3
"""winner_signature.py - Stage 1 of the "imitate unusually successful agents" idea.

The idea's loop is: run -> find unusually successful agents -> record what they saw and did -> train a
policy to imitate them -> repeat. Before paying for any imitation, this answers the cheaper question:

    IS THERE A BEHAVIOURAL SIGNATURE THAT SEPARATES WINNERS FROM LOSERS?

If yes, imitation (or just encoding the rule) is justified. If no, the whole ML track is falsified for
free. This reads recordings produced with `replay.py --traces` - no controller change, no training.

SELECTION (the part that must not be luck). Individual lifetime is contaminated by things an agent did
not choose: spawn position, a predator near at birth, and its INHERITED genome (max_energy, vision,
speed are heritable and observable). So:
  * primary selector = `descendants_alive_end` - how much of its line was still alive at the end. An
    agent whose line survives is the one whose behaviour AND genes actually worked.
  * `e_max` is reported separately as an INHERITED trait, not a behaviour: if success is explained by
    inherited max_energy, then imitation has little to offer and genome selection is the lever instead.

Usage:  ./winner_signature.py <replay_dir_or_files...>
"""
import glob
import gzip
import json
import os
import statistics as st
import sys

BEHAVIOURS = ["travel_per_fruit", "lockout_frac", "fruits_per_1k", "spawns", "sprint_frac",
              "obs_fruit_per_1k", "obs_pred_per_1k"]
# NOTE: wall_block / turned_from_fruit are recorded per DEATH event (see replay_survey.py), not per
# agent trace, so they are deliberately not compared here rather than silently reported as zeros.
INHERITED = ["e_max"]


def load(path):
    return json.load(gzip.open(path, "rt")) if path.endswith(".gz") else json.load(open(path))


def enrich(agents):
    out = []
    for a in agents:
        t = max(1, a["ticks"])
        b = dict(a)
        b["fruits_per_1k"] = 1000.0 * a["fruits"] / t
        b["sprint_frac"] = a.get("sprint_ticks", 0) / t
        b["obs_fruit_per_1k"] = 1000.0 * a.get("obs_fruit", 0) / t
        b["obs_pred_per_1k"] = 1000.0 * a.get("obs_pred", 0) / t
        b["wall_block"] = a.get("wall_block", 0)
        b["turned_from_fruit"] = a.get("turned_from_fruit", 0)
        if a["ticks"] < 100:            # ignore agents that never really lived: their rates are noise
            continue
        out.append(b)
    return out


def main():
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "*.json.gz")) if os.path.isdir(a) else [a])
    per_episode = []
    allw, alll = [], []
    for f in files:
        rec = load(f)
        if not rec.get("agents"):
            continue
        ag = enrich(rec["agents"])
        if len(ag) < 8:
            continue
        ag.sort(key=lambda a: -(a["descendants_alive_end"] * 1000 + a["descendants"]))
        k = max(3, len(ag) // 4)
        win, lose = ag[:k], ag[-k:]
        per_episode.append((rec["seed"], rec["final"]["ticks"], win, lose))
        allw += win
        alll += lose

    if not per_episode:
        print("no traced replays found - record with: replay.py corpus --traces ...")
        return

    print(f"{len(per_episode)} episodes with per-agent traces, {len(allw)} winners vs {len(alll)} losers\n")
    print(f"{'behaviour':>20} {'winners':>10} {'losers':>10} {'ratio':>7}  {'verdict':>28}")
    print("-" * 84)
    for k in BEHAVIOURS + INHERITED:
        w = [a[k] for a in allw if a.get(k) is not None]
        l = [a[k] for a in alll if a.get(k) is not None]
        if not w or not l:
            continue
        mw, ml = st.median(w), st.median(l)
        ratio = (mw / ml) if ml else float("inf")
        # Consistency ACROSS episodes: the direction must be taken from the pooled medians, not
        # hardcoded - otherwise a metric where winners are legitimately HIGHER reports "0/16".
        higher = mw > ml
        wins = 0
        for _sd, _t, win, lose in per_episode:
            ww = [a[k] for a in win if a.get(k) is not None]
            ll = [a[k] for a in lose if a.get(k) is not None]
            if ww and ll:
                dw = st.median(ww) - st.median(ll)
                if dw != 0 and (dw > 0) == higher:
                    wins += 1
        tag = "INHERITED trait" if k in INHERITED else "behaviour"
        consist = f"{wins}/{len(per_episode)} eps consistent, winners {('HIGHER' if higher else 'LOWER'):6s} ({tag})"
        print(f"{k:>20} {mw:>10.2f} {ml:>10.2f} {ratio:>7.2f}  {consist:>28}")

    print("\nverdict: a behaviour is a candidate IMITATION TARGET if winners differ consistently")
    print("across episodes (>=70% of episodes) AND its ratio is large; 'e_max' being decisive instead")
    print("would mean the lever is genome selection, not imitation.")


if __name__ == "__main__":
    main()
