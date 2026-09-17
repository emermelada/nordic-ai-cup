# Moving the endpoint to the VPS (latency fix)

## Why
The grader stops the game once **accumulated response wait hits 600 s**. Score ≈ `600 s ÷ per-request
latency`. We measured the current public path at **median ~80 ms / mean ~105 ms** (p90 163 ms) — almost
all of it the **cloudflared tunnel → home-Mac uplink** leg. Our endpoint itself answers in **0.8 ms**.
Hosting the origin in a datacenter puts it ~1–5 ms from the Cloudflare edge, so the same 600 s buys far
more ticks. Rough: 80 ms → ~7.5k ticks (~800); 30 ms → ~20k ticks (~2000); 20 ms → 30k cap (~3000).

## Build the image (on the VPS, or build here and copy)
From `survival-simulator/`:
```
docker build -f Dockerfile.vps -t nac-survival-vps .
```
Then run:
```
docker run -d --name nac-survival-vps -p 9052:9052 --restart unless-stopped nac-survival-vps
curl -s localhost:9052/            # expect {"message":...,"policy":"best_controller(potential-field, evolved)"}
curl -s localhost:9052/debug/stats
```

## Pick a routing option

### Option A — run the existing named tunnel ON the VPS (easiest; reuses survival.zaitzev.com)
Keeps TLS + routing exactly as-is; just moves the origin off the home uplink.
1. Copy the tunnel credentials + config to the VPS:
   `~/.cloudflared/8c03f8a7-0996-4c62-a4f3-11ba3922488d.json` and `~/.cloudflared/config.yml`
   (same content: ingress `survival.zaitzev.com -> http://localhost:9052`).
2. On the VPS: install cloudflared, then
   `cloudflared tunnel run nac-survival`  (systemd: `cloudflared service install`).
3. **Stop the tunnel on the Mac** so only the VPS connector serves:
   `pkill -f 'cloudflared tunnel run nac-survival'`.
4. Verify: `curl -s -o /dev/null -w '%{http_code}\n' https://survival.zaitzev.com/` → 200.

### Option B — Cloudflare proxied A record straight to the VPS (fastest)
No tunnel hop at all: grader → Cloudflare edge → VPS.
1. Cloudflare DNS: `survival.zaitzev.com` A → `<VPS public IP>`, **Proxied** (orange).
2. Cloudflare proxied origins only allow certain ports. Publish the container on an allowed HTTP port,
   e.g. `-p 8080:9052`, and set the CF origin to `http://<VPS IP>:8080`
   (SSL/TLS mode: Flexible, or Full if you front it with TLS on 443 via Caddy).
3. Verify as above.

### Option C — direct TLS on the VPS, no Cloudflare (lowest latency, more setup)
DNS A record **unproxied** (grey) → VPS IP; run Caddy/nginx on :443 with a Let's Encrypt cert and
reverse-proxy to `localhost:9052`. Grader → VPS directly.

## Measure before/after
```
bash experiments/_latency_probe.sh          # public-path latency, run from the Mac
```
Compare median/mean to the current ~80/105 ms. Expect Option A ≈ 30–50 ms, Option B/C ≈ 15–35 ms.

## Rollback
Just restart the Mac tunnel: `scripts/up.sh` (and stop the VPS tunnel/container if Option A).
The Mac service is untouched while you trial the VPS.

## Notes
- The Mac container (`nordic-ai-cup-survival-simulator-1`) stays as a working fallback until the VPS
  path is proven.
- `/app/predict_log.jsonl` on whichever host is live logs every grader step (use
  `experiments/analyze_grader_log.py` to decode it).
- Never rebuild/restart the live host while a validation attempt is queued.