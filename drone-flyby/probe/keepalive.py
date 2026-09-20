"""Keep the detector warm so frame 1 is not the slow one.

The first inference after a pause is markedly slower -- the handover records a
run that lost frames 2-5 right after a restart, and a cold first request is
already known to cost the opening frame. This sends one synthetic frame every
few seconds, which costs nothing on an idle GPU and keeps every CUDA kernel and
allocator arena hot.

It answers with a distinct sequence_id so it never disturbs a real attempt's
tracker state (flyby keeps per-sequence state), and it stands down entirely
while an attempt is in flight: a stray 90 ms of GPU work at the wrong moment is
exactly what turns into a stalled camera command, which is worth more score
than a cold start is.
"""
import base64, datetime, json, time, urllib.request
import numpy as np, cv2

URL = "http://localhost:10200/predict"
PERIOD = 15.0
LOG = "data/serve.log"
QUIET_FOR = 6.0      # seconds of no real traffic before warming again

# Flat grey: a noise image invents dozens of detections and makes the warm-up
# cost three times what a real frame costs.
img = np.full((540, 960, 3), 114, np.uint8)
ok, buf = cv2.imencode('.png', img)
payload = {
    "sequence_id": "keepalive", "frame": 1, "frame_index": 0,
    "request_id": "keepalive:0", "frame_interval_ms": 333, "response_timeout_ms": 3333,
    "original_width": 3840, "original_height": 2160,
    "view": {"resolution_level": 1, "center_x": 1920, "center_y": 540, "view_id": "k",
             "image": base64.b64encode(buf.tobytes()).decode(), "image_media_type": "image/png",
             "width": 960, "height": 540, "source_region_xyxy": [960, 0, 2880, 1080]},
    "camera_constraints": {"maximum_center_delta": 1102.0, "allowed_resolution_levels": [0, 1, 2],
                           "center_bounds": [{"resolution_level": 1, "width": 960, "height": 540,
                                              "minimum_center_x": 960, "maximum_center_x": 2880,
                                              "minimum_center_y": 540, "maximum_center_y": 1620}],
                           "full_view_reset_exempt_from_delta": True},
    "camera_command_feedback": None,
}
body = json.dumps(payload).encode()
def attempt_in_flight():
    """True if the service answered a real (non-warm-up) frame just now."""
    try:
        with open(LOG, 'rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 4096))
            lines = [l for l in handle.read().decode('utf-8', 'ignore').splitlines() if ' INFO frame ' in l]
        if not lines:
            return False
        last = lines[-1]
        if '(index 0)' in last:          # that is our own keepalive
            return False
        stamp = datetime.datetime.strptime(last.split(' INFO')[0], '%Y-%m-%d %H:%M:%S,%f')
        return (datetime.datetime.now() - stamp).total_seconds() < QUIET_FOR
    except Exception:
        return False


while True:
    if attempt_in_flight():
        time.sleep(2.0)
        continue
    try:
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        pass
    time.sleep(PERIOD)
