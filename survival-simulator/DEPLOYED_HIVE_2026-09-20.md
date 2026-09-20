# DEPLOYED: survival.zaitzev.com now serves hive (survival-v2)

**Executed 2026-09-20 ~07:45 UTC on the owner's explicit instruction.** The previous controller (the
heuristic `best_controller.py` 252f0ba1) was NOT deleted: it is stopped and restarts in one command.

## What changed
- Staged `/opt/surv/{hive.py,server.py,deploy/}` on the graded host, `hive.py` = sha256
  **829e4147fb309c5e08ef99ad5d25d811d0819f3c9c2db6a83210fa72e8f5e4a1** (the exact artifact measured at
  +37.6% over the incumbent, verified byte-identical before staging).
- smd unit `surv.service`: `/opt/nacv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 9052`
  (plain uvicorn, the same serving mode the container used), `Restart=always`, enabled at boot.
- Port handover: `docker stop nac-survival-vps` -> freed :9052 -> `systemctl start surv`.
- Container snapshot taken BEFORE the swap: image `nac-survival-vps:pre-hive-0920-0737`
  (id f97f722b0759), and the container itself is intact (`Exited (0)`).

## Verification before and after
- Before: official client driven against the staged instance on :9053 -> 3,049 requests, 0 errors,
  sim_time 300 s with 33 agents alive, decide 6.0-6.9 ms mean / 19.8 ms max.
- After: `https://survival.zaitzev.com/` 200 in 11 ms and returns hive's own health string;
  grader traffic (client IP `46.62.240.126`, the Hetzner grader address) has been served with
  **0 errors**; decide 3.8-5.3 ms mean, max 20.2 ms; service `active`, `NRestarts=0`.
- Note discovered live: the grader connects **directly to `<ip>:9052`**, not only through the hostname
  (its IP appears as a client in hive's own log). Both paths land on the same port, so the swap covers
  both; but it means any future test client aimed at that port competes with the graded attempt.

## Rollback (one command, ~5 s)
    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'systemctl stop surv; docker start nac-survival-vps; \
      sleep 3; curl -s -o /dev/null -w "public %{http_code} %{time_total}s\n" https://survival.zaitzev.com/'

To restore the exact previous image on a rebuilt container: `docker run ... nac-survival-vps:pre-hive-0920-0737`.

## Lesson recorded
Two clients must never drive one controller process: hive (like the container) keeps ONE internal state
and resets when `sim_time` goes backwards, so a test client running alongside a graded attempt makes both
games interleave and thrash that state. A test client was co-running for ~4 minutes on this swap and was
killed the moment the grader's IP appeared in the log; the in-flight attempt may be degraded, and the
board keeps the best attempt, so a fresh validation is the clean way to measure hive on the platform.
