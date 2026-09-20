# DEPLOY PLAN — swap the graded endpoint from the heuristic to hive (survival-v2)

**NOT EXECUTED.** Nothing about the production endpoint was changed by this session. This is the
one-command-at-a-time recipe for the owner's decision, with the evidence and the rollback.

## Why

Measured, same simulator, same 80 fresh paired seeds (`FINDING_HIVE_VS_SERVED_2026-09-20.md`):

    served best_controller 252f0ba1 :  8,110 ticks = 811 score
    hive (survival-v2) 829e4147     : 11,517 ticks = 1,152 score
    paired +3,406 +- 438 ticks, t = 7.78, W/L 65/15, p10 3,881 -> 7,139

hive's latency in the real sim: mean 4.09 ms, p99 11.7 ms = 29 s of the ~600 s grader budget.
Second independent confirmation block (40 more fresh seeds) was still running at hand-off.

## What is live now (verified)

- `survival.zaitzev.com` -> Caddy on 94.237.34.245: `reverse_proxy 127.0.0.1:9052`.
- `:9052` is held by `docker-proxy` for container `nac-survival-vps`
  (`uvicorn agent_server:app --host 0.0.0.0 --port 9052`), controller sha256 252f0ba1…
- hive's server: `uvicorn server:app --host 0.0.0.0 --port 9052` as systemd unit `surv`, files in
  `/opt/surv`, deps numpy/orjson/uvicorn. Source: `git show origin/survival-v2:survival-v2/{hive.py,server.py,deploy/*}`.

Because Caddy proxies to `127.0.0.1:9052` regardless of who listens there, the swap is a port handover
and the rollback is one command. **Never do this while a validation is in flight.**

## Pre-flight (both read-only)

    # is an attempt running? the log grows ~1 row per tick while one is
    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 \
      'docker exec nac-survival-vps sh -c "tail -1 /app/predict_log.jsonl; sleep 10; tail -1 /app/predict_log.jsonl"'
    # if the two lines differ, an attempt is LIVE - wait.

    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'curl -s -o /dev/null -w "%{http_code} %{time_total}\n" https://survival.zaitzev.com/'

## Step 1 - snapshot the incumbent (so rollback is trivial and byte-exact)

    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'docker commit nac-survival-vps nac-survival-vps:pre-hive-$(date -u +%m%d-%H%M) && docker images | head -3'
    # (existing backups inside the container: /root/code_backup_f90cb4e3.py, /root/params_backup_c6.json)

## Step 2 - stage hive on the graded box and smoke-test it on a SPARE port

    # from a checkout of the survival-v2 branch (or copy /opt/sv2/{hive.py,server.py,deploy} on box1):
    scp -i ~/.ssh/vps_hermes survival-v2/{hive.py,server.py} root@94.237.34.245:/opt/surv/
    scp -i ~/.ssh/vps_hermes survival-v2/deploy/setup_vm.sh survival-v2/deploy/requirements-server.txt \
        root@94.237.34.245:/opt/surv/deploy/
    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'PORT=9053 bash /opt/surv/deploy/setup_vm.sh'   # binds :9053, container untouched
    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'curl -s http://127.0.0.1:9053/api | head -c 200'

## Step 3 - hand over the port (the only step that touches the graded path)

    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'docker stop nac-survival-vps && systemctl restart surv && sleep 2 && ss -ltnp | grep 9052'

## Step 4 - verify before trusting it

    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'curl -s http://127.0.0.1:9052/api; echo; \
      curl -s -o /dev/null -w "public %{http_code} %{time_total}s\n" https://survival.zaitzev.com/'
    # then a real payload through the public path, and a 3-game rehearsal on the box before validating:
    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'journalctl -u surv -n 40 --no-pager | tail -20'

## Step 5 - validate

Trigger a board validation. The board keeps the best attempt, so a *worse* attempt costs nothing but
recovers no ground either. Expect the attempt mean to move from ~750-820 toward ~1,050-1,150 if the
local measurement transfers (it did transfer for the heuristic: local 811 vs official 731-819).

## Rollback (any time, ~5 s, byte-exact)

    ssh -i ~/.ssh/vps_hermes root@94.237.34.245 'systemctl stop surv; docker start nac-survival-vps; sleep 2; \
      curl -s -o /dev/null -w "public %{http_code} %{time_total}s\n" https://survival.zaitzev.com/'

## Two things I could not verify from here

1. `212.147.249.39` (`survival-v67.zaitzev.com`) serves *something* with `requests: 41920, errors: 0`, and
   the key I hold is rejected there. If the competition is registered to THAT endpoint rather than
   `survival.zaitzev.com`, this whole plan is unnecessary — confirm which hostname the board was
   registered with before spending the swap.
2. hive has never had a fair platform validation (its one tunnel attempt scored 320 through a
   Cloudflare tunnel with ~126 ms/tick, cut at t~500). Step 5 is therefore the first real test of
   transfer for it, which is exactly why the rollback is one command.
