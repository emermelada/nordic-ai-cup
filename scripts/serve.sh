#!/usr/bin/env bash
# Start a service + public tunnel, and VERIFY the whole chain end to end. Then optionally watch it.
#
#   scripts/serve.sh survival-simulator          start container + tunnel, verify, print URL
#   scripts/serve.sh medical-appointment
#   scripts/serve.sh drone-flyby
#   scripts/serve.sh --check                    re-verify the URL recorded by the last run
#   scripts/serve.sh --outside                  also verify from a third-party vantage point
#   scripts/serve.sh --dns                      explain/verify the local DNS situation
#   scripts/serve.sh --watch [--repair]         re-verify every 60s, --repair auto-restarts
#
# Why this exists: trycloudflare quick tunnels get revoked server-side without warning, and a
# sleeping Mac stops Docker Desktop's VM and the tunnel with it. "It worked ten minutes ago" is
# not evidence. This script only reports what the PUBLIC URL does right now.
#
# TWO TRAPS THIS SCRIPT DELIBERATELY WORKS AROUND (both cost real time on 2026-09-17):
#   1. This Mac runs Tailscale with "Use Tailscale DNS" enabled, and Tailscale's resolver
#      (100.100.100.100) fails to resolve *.trycloudflare.com even though the name IS published
#      globally. So a plain `curl https://x.trycloudflare.com` dies with "could not resolve host"
#      and looks like a dead tunnel. Fix: query PUBLIC dns (1.1.1.1) and use curl --resolve.
#   2. A brand-new quick-tunnel hostname resolves to DIFFERENT edge IPs from different resolvers
#      (e.g. 1.1.1.1 -> 104.16.230.132, 8.8.8.8 -> 104.16.231.132). Both work; don't misread it.
#
# New file, added 2026-09-17. Not part of the original scaffold - delete if unwanted.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# NOTE: macOS ships /bin/bash 3.2 (2007), which has NO associative arrays.
# So the maps are case statements, not `declare -A`. Same reason we avoid ${!arr[*]}.
SERVICES="medical-appointment drone-flyby survival-simulator"

# Host port each service is published on (left side of the compose "ports" mapping).
port_for() {
  case "$1" in
    medical-appointment) echo 8001 ;;
    drone-flyby)         echo 8002 ;;
    survival-simulator)  echo 9052 ;;   # the template's agent endpoint, see agent_server.py
    *) return 1 ;;
  esac
}

# Per-service probe contract, as "GETPATH|POSTPATH|POSTBODY".
# survival-simulator is NOT the old scaffold: it serves GET / and POST /predict, and /predict
# validates a full StepResponse, so a bare {} body would 422 (a false failure).
probe_for() {
  case "$1" in
    survival-simulator)
      echo '/|/predict|{"game_status":"ok","score":0.0,"sim_time":0.0,"n_agents":0,"agent_status":[]}' ;;
    medical-appointment|drone-flyby)
      echo '/api|/predict|{}' ;;
    *) return 1 ;;
  esac
}

# The permanent Cloudflare named-tunnel hostname per service, if one is set up.
# These never change, so unlike a quick tunnel they can be submitted once and forgotten.
named_host_for() {
  case "$1" in
    survival-simulator) echo survival.zaitzev.com ;;
    *) return 1 ;;
  esac
}

state_file() { echo "/tmp/nac-${1}.url"; }
log_file()   { echo "/tmp/nac-${1}-cloudflared.log"; }

# All diagnostics go to STDERR. That is deliberate: functions like start_tunnel return the URL
# on stdout via command substitution, so stdout must contain ONLY the value being returned.
say()  { printf '%s\n' "$*" >&2; }
ok()   { printf '  [OK]   %s\n' "$*" >&2; }
bad()  { printf '  [FAIL] %s\n' "$*" >&2; }
info() { printf '  ..     %s\n' "$*" >&2; }
warn() { printf '  [WARN] %s\n' "$*" >&2; }

# --named may appear anywhere on the command line; pull it out by scanning, so we avoid
# arrays (bash 3.2 chokes on expanding an empty array under `set -u`).
NAMED=no
for _a in "$@"; do [ "$_a" = "--named" ] && NAMED=yes; done

# ---------------------------------------------------------------- DNS reality

# Ask a resolver that is not this Mac's. Returns the A record, or empty.
public_ip() {
  dig +short @1.1.1.1 "$1" A 2>/dev/null | grep -E '^[0-9.]+$' | head -1
}

detect_dns_trap() {
  if command -v tailscale >/dev/null 2>&1; then
    if tailscale dns status 2>/dev/null | grep -q 'Tailscale DNS: enabled'; then
      warn "Tailscale is handling DNS on this Mac. It does NOT resolve *.trycloudflare.com,"
      warn "so plain curl/browser tests here can fail even when the tunnel is perfectly healthy."
      warn "This script uses public DNS + curl --resolve to get around it."
      warn "To fix it system-wide (your browser too), run either:"
      warn "    sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder"
      warn "  or  tailscale set --accept-dns=false"
    fi
  fi
}

# ---------------------------------------------------------------- verification

# The only check that matters: what does the public URL do from outside, right now?
verify() {
  local service=$1 url=$2 tries=${3:-1}
  local host=${url#https://} ip code body getpath postpath postbody
  IFS='|' read -r getpath postpath postbody <<< "$(probe_for "$service")"

  say "  public  $url"

  # Step 1: is the name published to the world? (this Mac's resolver is untrustworthy)
  ip=$(public_ip "$host")
  if [ -z "$ip" ]; then
    bad "no public A record for $host (asked 1.1.1.1) - tunnel not published, or revoked"
    return 1
  fi
  ok "published in public DNS -> $ip"

  # Step 2: real request over the public path. --resolve replaces only the DNS lookup, so
  # traffic still goes out to Cloudflare's edge exactly as a grader's request would.
  for ((i=1; i<=tries; i++)); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
           --resolve "$host:443:$ip" "$url$getpath")
    [ "$code" = "200" ] && break
    [ "$i" -lt "$tries" ] && sleep 10
  done
  if [ "$code" = "200" ]; then ok "GET  $getpath  200"; else bad "GET  $getpath  $code"; fi

  # Step 3: the POST the organisers will actually make, with a VALID body, and assert the
  # response shape - a 200 with the wrong body is still a losing submission.
  body=$(curl -s --max-time 60 --resolve "$host:443:$ip" \
         -X POST "$url$postpath" -H 'content-type: application/json' -d "$postbody" 2>/dev/null)
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 --resolve "$host:443:$ip" \
         -X POST "$url$postpath" -H 'content-type: application/json' -d "$postbody")
  if [ "$code" = "200" ]; then ok "POST $postpath 200"; else bad "POST $postpath $code"; fi
  case "$body" in
    *'"actions"'*) ok "response body contains \"actions\"" ;;
    *)             bad "response body missing \"actions\": $(printf '%s' "$body" | head -c 120)" ;;
  esac

  [ "$code" = "200" ]
}

# Third-party vantage point: fetches the URL from a server that is not you and not your network.
outside_check() {
  local service=$1 url=$2 code
  local getpath; getpath=$(cut -d'|' -f1 <<< "$(probe_for "$service")")
  info "asking an external service to fetch the URL (proves it is reachable from the internet)..."
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
         "https://r.jina.ai/${url}${getpath}")
  if [ "$code" = "200" ]; then
    ok "outside-in fetch 200 (reached your service from a third-party network)"
  else
    bad "outside-in fetch $code (external fetcher could not reach it)"
  fi
  [ "$code" = "200" ]
}

# ---------------------------------------------------------------- start tunnel

start_tunnel() {
  local service=$1 port=$2 log; log=$(log_file "$service")
  pkill -f "cloudflared tunnel --url http://localhost:${port}" 2>/dev/null
  sleep 1
  : > "$log"
  nohup cloudflared tunnel --url "http://localhost:${port}" >"$log" 2>&1 &
  disown 2>/dev/null

  info "waiting for cloudflared to publish a URL..."
  local url="" i
  for i in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1)
    [ -n "$url" ] && break
    sleep 2
  done
  if [ -z "$url" ]; then
    bad "cloudflared published no URL in 60s - check $log"
    return 1
  fi
  ok "tunnel URL: $url"
  echo "$url" > "$(state_file "$service")"

  # Wait for PUBLICATION, judged by a public resolver - not by this Mac's broken one.
  info "waiting for the name to appear in public DNS (up to 90s)..."
  local host=${url#https://}
  for i in $(seq 1 18); do
    [ -n "$(public_ip "$host")" ] && { ok "public DNS live after ~$((i*5))s"; break; }
    sleep 5
  done
  echo "$url"
}

# ---------------------------------------------------------------- subcommands

cmd_up() {
  local service=$1 port probe
  if ! port=$(port_for "$1"); then
    bad "unknown service '$service'. known: $SERVICES"
    return 1
  fi
  probe=$(probe_for "$service")
  say "== $service (host port $port, contract: $probe) =="

  detect_dns_trap
  say ""

  info "starting container..."
  if ! docker compose up -d --build "$service" >/tmp/nac-build.log 2>&1; then
    bad "docker compose up failed - see /tmp/nac-build.log"
    return 1
  fi
  ok "container started"

  info "waiting for the healthcheck..."
  local health="" i
  for i in $(seq 1 20); do
    health=$(docker inspect "nordic-ai-cup-${service}-1" \
             --format '{{.State.Health.Status}}' 2>/dev/null)
    [ "$health" = "healthy" ] && break
    sleep 3
  done
  [ "$health" = "healthy" ] && ok "container healthy" || info "container health: ${health:-unknown}"

  # Prove the content is right BEFORE exposing it: what the container serves on localhost.
  local getpath; getpath=$(cut -d'|' -f1 <<< "$probe")
  local localcode; localcode=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
                               "http://localhost:${port}${getpath}")
  [ "$localcode" = "200" ] && ok "container answers locally on :$port$getpath" \
                           || bad "container does NOT answer locally ($localcode)"

  local url; url=$(start_tunnel "$service" "$port") || return 1

  say ""
  say "== verifying the public chain =="
  if verify "$service" "$url" 6 && outside_check "$service" "$url"; then
    say ""
    say "================================================================"
    say "  READY. This is the URL to submit:"
    say ""
    say "     $url"
    say ""
    say "  Re-verify before EVERY submission:"
    say "     scripts/serve.sh --check"
    say "================================================================"
    return 0
  fi
  say ""
  bad "the public chain is not healthy. Re-run this script for a fresh tunnel."
  return 1
}

cmd_check() {
  local service="" url="" f s nh
  if [ "$NAMED" = "yes" ]; then
    # Verify the permanent named-tunnel hostname instead of a throwaway quick tunnel.
    for s in $SERVICES; do
      if nh=$(named_host_for "$s"); then service=$s; url="https://$nh"; fi
    done
    if [ -z "$url" ]; then
      bad "no permanent hostname configured. Add one to named_host_for() in this script."
      return 1
    fi
    say "== permanent URL: $service =="
  else
    for s in $SERVICES; do
      f=$(state_file "$s")
      [ -f "$f" ] && { service=$s; url=$(cat "$f"); }
    done
    if [ -z "$url" ]; then
      bad "no recorded URL. Run: scripts/serve.sh <service>, or use --check --named"
      return 1
    fi
    say "== last known quick tunnel: $service =="
  fi
  verify "$service" "$url" 1 && outside_check "$service" "$url"
}

cmd_watch() {
  local repair=${1:-no} service url round=0
  service=$(for s in $SERVICES; do [ -f "$(state_file "$s")" ] && echo "$s"; done | tail -1)
  [ -z "$service" ] && { bad "nothing to watch. Run: scripts/serve.sh <service>"; return 1; }
  say "watching $service (repair=$repair). Ctrl-C to stop."
  while true; do
    round=$((round+1))
    url=$(cat "$(state_file "$service")")
    printf '[%s] round %d: ' "$(date +%H:%M:%S)" "$round"
    if verify "$service" "$url" 1 >/tmp/nac-verify.out 2>&1; then
      printf 'OK\n'
    else
      printf 'DOWN\n'
      sed 's/^/    /' /tmp/nac-verify.out
      if [ "$repair" = "--repair" ]; then
        info "attempting repair: restarting container + tunnel"
        cmd_up "$service" || info "repair failed"
      fi
    fi
    sleep 60
  done
}

cmd_dns() {
  local h=${1:-}
  say "== DNS reality check =="
  detect_dns_trap
  if [ -z "$h" ]; then
    say ""
    say "Pass a hostname to test both resolvers, e.g.:"
    say "  scripts/serve.sh --dns my-tunnel.trycloudflare.com"
    return 0
  fi
  say ""
  say "  host: $h"
  say "    public (@1.1.1.1) : $(public_ip "$h")"
  say "    local  (this Mac) : $(dig +short "$h" A 2>/dev/null | head -1)"
  say "    curl              : $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "https://$h/" 2>&1)"
}

usage() {
  cat <<'EOF'
scripts/serve.sh <service>       start container + tunnel, verify publicly, print the URL
                                 services: medical-appointment (8001), drone-flyby (8002),
                                           survival-simulator (9052)
scripts/serve.sh --check         re-verify the URL recorded by the last run
scripts/serve.sh --check --named verify the PERMANENT named-tunnel URL instead
                                 (survival-simulator -> https://survival.zaitzev.com)
scripts/serve.sh --outside       with --check, also fetch from a third-party network
scripts/serve.sh --dns [host]    show whether this Mac's DNS is the thing lying to you
scripts/serve.sh --watch [--repair]
                                 re-verify every 60s; --repair restarts container+tunnel on failure

Quick tunnels: only needed when no named tunnel exists. Prefer the named tunnel
(scripts/up.sh) - the URL is permanent, so you submit it once and never resubmit.

Verification asks PUBLIC dns (1.1.1.1) and uses curl --resolve, because this Mac's Tailscale
resolver cannot see *.trycloudflare.com. Quick tunnels also get revoked silently - never assume
"it worked ten minutes ago".
EOF
}

case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  --check)   shift; cmd_check ;;
  --named)   usage; exit 0 ;;
  --outside) shift; cmd_check ;;
  --dns)     shift; cmd_dns "${1:-}" ;;
  --watch)   shift; cmd_watch "${1:-no}" ;;
  *)         cmd_up "$1" ;;
esac
