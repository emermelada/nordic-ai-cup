"""deploy_gate.py — apply HARD RULE 1 to an arm from the results cache.

Usage: deploy_gate.py ARM [SEED_LO SEED_HI]
Reports paired score-points vs the injected BASE arm: n, mean, win%, p10 of the paired
diff (floor), and a bootstrap P(a 3-run validation nets a loss).
Score units: official points == 'score' field  (score = ticks/10 + fruit_energy/1000).
"""
import json, sys, collections, random, statistics

ARM = sys.argv[1]
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

rows = []
for l in open("/opt/nac/results/cache.jsonl"):
    try:
        j = json.loads(l)
    except Exception:
        continue
    if j.get("score") is not None and "seed" in j and "cand" in j:
        rows.append((j["cand"], int(j["seed"]), float(j["score"])))

by = collections.defaultdict(dict)
for c, s, sc in rows:
    by[c][s] = sc

base = by["BASE"]
common = sorted(s for s in by[ARM] if s in base and LO <= s <= HI)
d = [by[ARM][s] - base[s] for s in common]
if len(d) < 5:
    sys.exit(f"{ARM}: only {len(d)} paired seeds in {LO}-{HI}")

w = sum(1 for x in d if x > 1e-9); ls = sum(1 for x in d if x < -1e-9)
sd = statistics.pstdev(d)
mean = statistics.mean(d)
srt = sorted(d)
p10 = srt[max(0, int(0.10 * len(srt)) - 1)]
print(f"ARM {ARM}  seeds {common[0]}-{common[-1]}  n={len(d)}")
print(f"  mean paired {mean:+8.1f} pts   median {statistics.median(d):+8.1f}   sd {sd:6.1f}")
print(f"  win {w}/{len(d)} = {100*w/len(d):.0f}%   loss {ls}   tie {len(d)-w-ls}")
print(f"  floor p10 of paired diff {p10:+8.1f} pts   worst {srt[0]:+.0f}   best {srt[-1]:+.0f}")
random.seed(0)
bad = 0
for _ in range(4000):
    if sum(random.choice(d) for _ in range(3)) / 3 < 0:
        bad += 1
print(f"  P(mean-of-3 validation nets a LOSS vs BASE) = {100*bad/4000:.0f}%")
ok = (len(d) >= 40) and (w/len(d) >= 0.55) and (p10 >= 0)
print(f"  RULE 1: n>=40 {'OK' if len(d)>=40 else 'FAIL'} | win>=55% {'OK' if w/len(d)>=0.55 else 'FAIL'} | floor>=0 {'OK' if p10>=0 else 'FAIL'}  -> {'QUALIFIED' if ok else 'NOT QUALIFIED'}")
