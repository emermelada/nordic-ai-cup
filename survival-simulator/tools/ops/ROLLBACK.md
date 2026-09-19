# ROLLBACK / FAILOVER RUNBOOK

**Both original fallback boxes are gone** (212.147.239.222 and 94.237.81.173 no longer answer ping or
SSH), so until the standby below existed there was NO recovery path: one outage of the serving box meant
zero score permanently. This is that path, and it has been tested rather than assumed.

## Current layout

| role | host | state |
|---|---|---|
| **PRIMARY (graded)** | `94.237.34.245` (32 vCPU, Helsinki) | Caddy + `nac-survival-vps` on :9052, answers `https://survival.zaitzev.com`, serving controller sha256 `f90cb4e3…` |
| **STANDBY (tested)** | `212.147.236.122` (64 vCPU, Helsinki) | image built, container `nac-survival-STANDBY` on :9053 **created+stopped**, Caddy installed with a **validated** config, service **inactive and disabled** |
| experiments | same 64 vCPU box | tmux lanes (`betB`, `corpus1..4`) — these MUST be stopped if the standby ever serves |

The standby was verified live on 2026-09-19: started on :9053, `GET /` → 200, a real `POST /predict`
returned valid actions, in-container `best_controller.py` sha256 = `f90cb4e3…` (identical to primary),
then stopped again. Caddy's config validates and it is disabled, so it makes no ACME attempts while DNS
points elsewhere.

## Trigger

Primary unreachable (SSH or HTTP), or serving returning errors. Check first:
`curl -s -o /dev/null -w '%{http_code} %{time_total}\n' https://survival.zaitzev.com/`

## Failover — target under 5 minutes, only step 3 needs the human

```bash
# 1. free the cores so serving is not competing with experiments (~10 s)
ssh -i ~/.ssh/vps_hermes root@212.147.236.122 \
  'for s in betB corpus1 corpus2 corpus3 corpus4; do tmux kill-session -t $s 2>/dev/null; done; pkill -f sched.py; pkill -f evolve_meta.py'

# 2. start the standby and prove it locally BEFORE touching DNS (~10 s)
ssh -i ~/.ssh/vps_hermes root@212.147.236.122 \
  'docker start nac-survival-STANDBY; sleep 5; curl -s -o /dev/null -w "local %{http_code}\n" localhost:9053/'

# 3. HUMAN STEP: Cloudflare DNS — A  survival  ->  212.147.236.122   (proxy OFF / grey)
#    verify it propagated before step 4 or the ACME challenge fails:
dig +short survival.zaitzev.com A @1.1.1.1

# 4. start Caddy (it obtains the certificate; needs step 3 propagated) (~30 s)
ssh -i ~/.ssh/vps_hermes root@212.147.236.122 'systemctl start caddy; sleep 20; journalctl -u caddy -n 20 | grep -i "certificate obtained"'

# 5. verify from outside, and confirm the traffic is landing on the standby
curl -s https://survival.zaitzev.com/
curl -s -o /dev/null -w 'public %{http_code} in %{time_total}s\n' -X POST https://survival.zaitzev.com/predict \
  -H 'Content-Type: application/json' -d '{"game_status":"RUNNING","score":0.0,"agent_status":[]}'
ssh -i ~/.ssh/vps_hermes root@212.147.236.122 'curl -s localhost:9053/debug/stats'   # log_lines increments

# 6. protect serving latency on the shared box: keep experiments OFF while it serves
```

## Reverting (standby back to primary)

Point DNS `survival` back to `94.237.34.245`, stop Caddy and the container on the standby
(`systemctl stop caddy; docker stop nac-survival-STANDBY`), and confirm the primary answers.

## Invariants that must hold

* The standby must serve the **same controller hash** as the primary. After any deploy to the primary,
  rebuild the standby, or the "rollback" silently ships a different policy:
  `docker cp nac-survival-vps:/app/best_controller.py …` from the primary, then rebuild here.
* Never start Caddy on the standby while DNS points elsewhere — a failed ACME challenge is wasted
  Let's Encrypt rate limit, and repeated failures can lock the domain out.
* Experiments and serving on one box is an accepted trade for compute, but ONLY with the latency guard
  active (`tools/ops/latency_guard.sh`); `score ≈ min(ticks, 600 s / latency)` means a saturated CPU
  costs score directly.
