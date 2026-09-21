"""Turn probe scores into the numbers we actually want.

The grader returns one number per run: the mean of AP@0.50 over the classes
PRESENT in the ground truth. So a run that answers for one class only returns
AP_c / K, and a run that answers a single certain box returns m / (101 K),
where m is a small integer set by how many instances that class has. K, the
number of classes in the validation truth, follows from the second kind of run
and unlocks every other one.

    python probe/read_probe.py --kprobe 0.000707
    python probe/read_probe.py --k 14 --score mined3d_all=0.61 --score tank=0.031
"""

import argparse


def solve_k(score, k_range=(8, 20), m_range=(1, 6)):
    """Candidate (K, m) for a single-box probe scoring `score`."""
    out = []
    for k in range(*k_range):
        for m in range(*m_range):
            predicted = m / (101.0 * k)
            if abs(predicted - score) <= max(1e-9, 0.02 * predicted):
                out.append((k, m, predicted))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kprobe', type=float, help='score of the single-box probe')
    ap.add_argument('--k', type=int, help='classes present in the truth, if known')
    ap.add_argument('--score', action='append', default=[], help='name=score for a probe run')
    a = ap.parse_args()

    if a.kprobe is not None:
        print(f'single-box probe = {a.kprobe:.8f}')
        cands = solve_k(a.kprobe)
        if not cands:
            print('  no (K, m) fits. If the score is 0 the box was not a true positive;')
            print('  otherwise the class has few instances and m is larger than expected.')
        for k, m, predicted in cands:
            print(f'  K = {k:2d} classes, m = {m} (predicted {predicted:.8f})')
            print(f'      -> that class has roughly {100 // m if m else 0}-{100 // max(1, m - 1)} instances')

    if a.k and a.score:
        print(f'\nwith K = {a.k}:')
        total = 0.0
        for item in a.score:
            name, _, value = item.partition('=')
            ap_c = float(value) * a.k
            total += float(value)
            print(f'  {name:24s} score {float(value):.5f}  ->  AP = {ap_c:.4f}')
        print(f'  {"sum of runs":24s} score {total:.5f}  ->  mean AP = {total * a.k / max(1, len(a.score)):.4f}')


if __name__ == '__main__':
    main()
