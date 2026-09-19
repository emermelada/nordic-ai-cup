#!/usr/bin/env python3
"""budget.py - mid-game energy budget: production vs consumption vs standing stock, by phase.

Production is MEASURED by mass balance, not assumed:
    fruit energy produced = (change in standing stock) + (fruit energy the fleet absorbed)
Sustainable fleet size per phase = production per 1k ticks / energy need per agent per 1k ticks.
If actual N exceeds sustainable N in the mid-game, the fleet is overshooting its food supply.
"""
import gzip, json, os, sys

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def main(paths):
    per = {}
    for p in paths:
        rec = load(p)
        frames, ledger = rec["frames"], rec.get("ledger") or []
        if not ledger:
            print(f"{os.path.basename(p)}: no ledger, skipped")
            continue
        step = max(1, ledger[1][0] - ledger[0][0]) if len(ledger) > 1 else 10
        win = max(2, 1000 // step)
        import bisect
        ft = [f["t"] for f in frames]

        def stock_at(t):
            # frames are sampled every `every` ticks, so take the nearest sampled frame <= t
            i = bisect.bisect_right(ft, t) - 1
            if i < 0:
                i = 0
            return sum(x[2] for x in frames[i]["fruits"])

        for i in range(0, len(ledger) - win, win):
            blk = ledger[i:i + win]
            t0, t1 = blk[0][0], blk[-1][0]
            s0, s1 = stock_at(t0), stock_at(t1)
            if s0 is None or s1 is None:
                continue
            N = sum(r[1] for r in blk) / len(blk)
            income, prev = 0.0, None
            for r in blk:
                if prev is not None:
                    d = r[3] - prev
                    if d > 0.11:
                        income += (d - 0.1) * 1000.0
                prev = r[3]
            move = sum(r[4] for r in blk)
            need = 100.0 * N + move            # metabolism (0.1/tick) + commanded movement
            ph = (t0 // 3000) * 3
            per.setdefault(ph, []).append((N, s1, income, (s1 - s0), need, sum(r[5] for r in blk)))

    print(f"{'phase':>8} {'wins':>5} {'N':>6} {'stock':>8} {'dStock':>8} {'eaten':>8} {'produced':>9} {'need/agent':>11} {'sustainN':>9} {'spawns':>7}")
    for ph in sorted(per):
        rows = per[ph]
        n = len(rows)
        N = sum(r[0] for r in rows) / n
        stock = sum(r[1] for r in rows) / n
        eaten = sum(r[2] for r in rows) / n
        dst = sum(r[3] for r in rows) / n
        need = sum(r[4] for r in rows) / n
        prod = dst + eaten
        na = need / max(1.0, N)
        print(f"{ph:>4}-{ph+3}k {n:5d} {N:6.1f} {stock:8.0f} {dst:8.0f} {eaten:8.0f} {prod:9.0f} {na:11.1f} {prod/max(1e-9,na):9.1f} {sum(r[5] for r in rows)/n:7.1f}")

if __name__ == "__main__":
    main(sys.argv[1:])
