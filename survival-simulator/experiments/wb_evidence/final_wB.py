"""Final analysis for the wB thin-relay sweep. Reference = B0_ref (EXACT H1 params, evaluated FIRST as
a candidate in the SAME process as the arms), because nac-B1 was SIGKILLed (exit 137) during its trailing
LIVE run. B0_ref is a strictly better paired control: identical params, same process, and it costs no
arm position. nac-B2 also carries a complete LIVE, so the noise floor of the harness is measured as
B0_ref-vs-LIVE, same process, same params."""
import json
import statistics as st

def load(path):
    res, fruits = {}, {}
    for ln in open(path):
        p = ln.split()
        if "MEAN" in p and "[" in ln:
            mi = p.index("MEAN")
            lab = " ".join(p[:mi]) if mi > 1 else p[0]
            seeds = [int(x) for x in ln.split("[")[1].rstrip("]\n").split(",")]
            res[lab] = seeds
            fruits[lab] = float(p[mi + 5])   # MEAN <val> ticks | fruits <val> | seeds
    return res, fruits

B1, F1 = load("/tmp/wB/logs/B1.log")   # B0_ref, B1, B2, B3   (LIVE lost to SIGKILL)
B2, F2 = load("/tmp/wB/logs/B2.log")   # B0_ref, B1, B4, LIVE

# seeds are evaluated in the same order (1300..1319) in both containers -> index == seed
SEEDS = [1300 + i for i in range(20)]

print("=== noise floor (harness, same process) ===")
d = [B2["B0_ref"][i] - B2["LIVE (deployed)"][i] for i in range(20)]
print(f"  B0_ref vs LIVE  mean {st.fmean(d):+8.1f}  median {st.median(d):+8.1f}  min {min(d):+7.0f}  max {max(d):+7.0f}"
      f"  seeds>500: {sum(1 for x in d if x > 500)}")
d = [B1["B0_ref"][i] - B2["B0_ref"][i] for i in range(20)]
print(f"  B0_ref(B1 proc) vs B0_ref(B2 proc)  mean {st.fmean(d):+8.1f}  median {st.median(d):+8.1f}"
      f"  min {min(d):+7.0f}  max {max(d):+7.0f}   (cross-process variance)")

out = {}
for proc, tabs, frt in [("B1", B1, F1), ("B2", B2, F2)]:
    ref = tabs["B0_ref"]
    for lab in tabs:
        if lab == "B0_ref":
            continue
        v = tabs[lab]
        dv = [v[i] - ref[i] for i in range(20)]
        row = dict(proc=proc, mean=st.fmean(v), median=st.median(v), min=min(v),
                   delta_mean=st.fmean(dv), delta_median=st.median(dv),
                   seeds_won_500=sum(1 for x in dv if x > 500),
                   seeds_lost_500=sum(1 for x in dv if x < -500),
                   n_better=sum(1 for x in dv if x > 0), n_worse=sum(1 for x in dv if x < 0),
                   fruits=frt[lab], fruits_ref=frt["B0_ref"], per_seed_delta=dv)
        out[f"{proc}:{lab}"] = row
        print(f"\n{proc}:{lab}  mean {row['mean']:8.1f}  median {row['median']:8.1f}  min {row['min']:6.0f}"
              f"   dVsB0ref mean {row['delta_mean']:+8.1f} median {row['delta_median']:+8.1f}"
              f"   wins>500 {row['seeds_won_500']:2d}/20  losses<-500 {row['seeds_lost_500']:2d}/20"
              f"   better {row['n_better']}/20")
        print(f"   fruits {row['fruits']:.1f} vs ref {row['fruits_ref']:.1f}")

# dose-response on thinning target, same trigger (3000) and same process (B2)
print("\n=== dose-response (trigger 3000, vis 0.55, process B2) ===")
print(f"  target 2 (B1) delta {out['B2:B1']['delta_mean']:+8.1f}   target 6 (B4) delta {out['B2:B4']['delta_mean']:+8.1f}")

json.dump(out, open("/tmp/wB/wB_final.json", "w"), indent=1)
print("\nwrote /tmp/wB/wB_final.json")
