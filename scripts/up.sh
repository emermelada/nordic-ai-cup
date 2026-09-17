#!/usr/bin/env bash
# One command to get the survival-simulator agent endpoint permanently live AND keep the Mac awake.
#
#   scripts/up.sh                        start container, verify the permanent URL, then run the
#                                        named Cloudflare tunnel under caffeinate.
#                                        Ctrl-C stops the tunnel and lets the Mac sleep again.
#   scripts/up.sh --probe                do the checks and EXIT, without starting the tunnel
#                                        (useful to confirm the tunnel is already running)
#
# Why caffeinate: a validation/evaluation attempt calls your URL whenever the organisers' queue
# reaches you. If the Mac sleeps mid-run on an evaluation attempt (one try only, 3 runs averaged)
# you lose it. caffeinate keeps the machine awake for exactly as long as this script runs.
#
# Setup this script assumes (one-time, already done):
#   ~/.cloudflared/cert.pem                 from `cloudflared tunnel login`
#   ~/.cloudflared/config.yml               maps survival.zaitzev.com -> http://localhost:9052
#   cloudflared tunnel nac-survival         the named tunnel those two refer to
#
# New file, added 2026-09-17. Not part of the original scaffold - delete if unwanted.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

SERVICE="survival-simulator"
PORT="9052"
HOST="survival.zaitzev.com"
TUNNEL="nac-survival"
CONTAINER="nordic-ai-cup-${SERVICE}-1"
URL="https://${HOST}"
GETPATH="/"

PROBE_ONLY=no
[ "${1:-}" = "--probe" ] && PROBE_ONLY=yes

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [OK]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
info() { printf '  ..     %s\n' "$*"; }

say "== $SERVICE -> $URL (port $PORT) =="

# ---------------------------------------------------------------- preflight
if ! command -v cloudflared >/dev/null 2>&1; then
  bad "cloudflared not on PATH"; exit 1
fi
CFG="$HOME/.cloudflared/config.yml"
if [ ! -f "$CFG" ]; then
  bad "missing $CFG - create it with: cloudflared tunnel login, then cloudflared tunnel create $TUNNEL"
  exit 1
fi
if ! grep -q "$HOST" "$CFG"; then
  info "note: $CFG does not mention $HOST - the tunnel may point somewhere else"
fi
ok "cloudflared + config present"

# ---------------------------------------------------------------- 1. container
info "starting container..."
if ! docker compose up -d "$SERVICE" >/tmp/nac-up-compose.log 2>&1; then
  bad "docker compose up failed - see /tmp/nac-up-compose.log"
  exit 1
fi
health=""
for _i in $(seq 1 20); do
  health=$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null)
  [ "$health" = "healthy" ] && break
  sleep 3
done
if [ "$health" = "healthy" ]; then ok "container healthy"; else bad "container health: ${health:-unknown}"; fi

# ---------------------------------------------------------------- 2. local
localcode=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://localhost:${PORT}${GETPATH}")
if [ "$localcode" = "200" ]; then ok "local :$PORT$GETPATH 200"; else bad "local :$PORT$GETPATH $localcode"; fi

# ---------------------------------------------------------------- 3. permanent public URL
if pgrep -f "cloudflared tunnel run ${TUNNEL}" >/dev/null 2>&1; then
  ok "tunnel '${TUNNEL}' is already running"
  TUNNEL_UP=yes
else
  TUNNEL_UP=no
fi

info "checking $URL ..."
code=""
for _i in $(seq 1 6); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$URL$GETPATH" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
if [ "$code" = "200" ]; then
  ok "public $URL$GETPATH 200"
elif [ "$TUNNEL_UP" = "no" ]; then
  info "public URL sad ($code) - expected, the tunnel is not running yet; starting it below"
else
  bad "public URL answered $code while a tunnel is running - check ~/.cloudflared/config.yml"
fi

if [ "$PROBE_ONLY" = "yes" ]; then
  say ""
  say "--probe: not starting anything. Re-check any time with:"
  say "   bash $REPO/scripts/serve.sh --check --named"
  [ "$code" = "200" ] && exit 0 || exit 1
fi

# ---------------------------------------------------------------- 4. tunnel + keepawake
# Retire any stale QUICK tunnel: the named one supersedes it, and two tunnels to one port
# just makes it ambiguous which URL is live.
pkill -f "cloudflared tunnel --url" 2>/dev/null && info "stopped a stale quick tunnel"

if [ "$TUNNEL_UP" = "yes" ]; then
  bad "tunnel '${TUNNEL}' is ALREADY running - refusing to start a second one"
  info "it is probably in another terminal. To restart it:"
  info "   pkill -f 'cloudflared tunnel run ${TUNNEL}'   then re-run this script"
  info "caffeinate does NOT cover that older process, so if you want it kept awake, restart it here."
  exit 1
fi

say ""
say "=================================================================="
say "  Starting '${TUNNEL}'. This terminal stays BUSY and the Mac stays"
say "  AWAKE until you press Ctrl-C."
say ""
say "  Live URL (permanent, submit once):   $URL"
say "  Re-check from another terminal:      bash $REPO/scripts/serve.sh --check --named"
say ""
say "  Keep the lid OPEN and the Mac plugged in."
say "=================================================================="
say ""

exec caffeinate -dims cloudflared tunnel run "$TUNNEL"
