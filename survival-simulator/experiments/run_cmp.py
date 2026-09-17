import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import multi_eval, summarize, print_summary, failure_analysis
from policies import heuristic_policy
from best_controller import make_policy, DEFAULT_PARAMS

H = 3000
pol = make_policy(dict(DEFAULT_PARAMS))

for name, fn in [("heuristic", heuristic_policy), ("potential", pol)]:
    r = multi_eval(fn, seeds=[100, 200, 300, 400, 500], horizon=H)
    s = summarize(r, name)
    print_summary(s)
    fa = failure_analysis(fn, seed=100, horizon=4000)
    print("   FA seed100:", fa["deaths"], "births:%d peak:%d fruit:%.0f predPen:%.1f score:%.0f final:%d collapse_ticks:%d" % (
        fa["births"], fa["peak"], fa["fruit_score"], fa["pred_penalty"], fa["score"], fa["final_agents"], fa["collapse_ticks"]))