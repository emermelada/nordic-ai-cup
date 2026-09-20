#!/usr/bin/env python3
"""queue_validations.py - drive the platform's validation queue back-to-back on the deployed hive.

Measures the thing that actually gets graded: the score distribution of the deployed controller as the
organisers' own simulator produces it. Sequenced ONE AT A TIME on purpose - two validations in flight
would interleave on a single hive process, and hive resets its state whenever sim_time goes backwards,
so parallel attempts corrupt each other.

Never prints the token. Writes one JSON line per finished attempt.

  python3 queue_validations.py --url https://survival.zaitzev.com/predict --n 12 \
      --out experiments/platform_runs/hive_validations.jsonl
"""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.request

API = "https://cases.nordicaicup.com/api/v1/usecases/survival-simulator"


def call(path, token, payload=None):
    url = "%s/%s" % (API, path)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET")
    req.add_header("x-token", token)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def status(token):
    return call("status", token)


def finished_set(st):
    out = {}
    for v in (st.get("validations") or []):
        a = v.get("attempt") or {}
        if a.get("finished_at"):
            out[a["finished_at"]] = {"score": a.get("score"), "submitted_at": a.get("submitted_at"),
                                     "started_at": a.get("started_at"), "errors": a.get("errors"),
                                     "service_url": a.get("service_url")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://survival.zaitzev.com/predict")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-wait", type=int, default=3600, help="seconds to wait for one attempt")
    args = ap.parse_args()
    token = os.environ.get("NAC_TOKEN")
    if not token:
        print("NAC_TOKEN env var missing", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = finished_set(status(token))
    print("[queue] %d validations already finished; target %d new" % (len(done), args.n), flush=True)
    for k in range(args.n):
        try:
            resp = call("validate/queue", token, {"url": args.url})
        except Exception as e:
            print("[queue] enqueue failed: %r; retrying in 60 s" % (e,), flush=True)
            time.sleep(60)
            continue
        uuid = resp.get("queued_attempt_uuid")
        print("[queue] %d/%d queued uuid=%s pos=%s" % (k + 1, args.n, uuid, resp.get("position_in_queue")), flush=True)
        t0 = time.time()
        got = None
        key = None
        while time.time() - t0 < args.max_wait:
            time.sleep(args.poll)
            try:
                st = status(token)
            except Exception as e:
                print("[queue] status failed: %r" % (e,), flush=True)
                continue
            now = finished_set(st)
            new = [key for key in now if key not in done]
            if new:
                key = sorted(new)[0]
                got = now[key]
                done = now
                break
        if got is None:
            print("[queue] attempt did not finish within %d s; moving on" % args.max_wait, flush=True)
            continue
        rec = {"queued_uuid": uuid, "service_url": args.url,
               "finished_at": key, "score": got["score"], "errors": got.get("errors"),
               "submitted_at": got.get("submitted_at"), "started_at": got.get("started_at")}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[queue] FINISHED score=%s errors=%s (written)" % (got["score"], got.get("errors")), flush=True)
        time.sleep(5)
    print("[queue] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
