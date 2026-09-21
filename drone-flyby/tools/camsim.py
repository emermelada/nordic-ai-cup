"""Walk a camera pattern for a whole flight and check every move is legal.

A refused command costs a frame of steering, and the evaluator applies its own
three checks; this runs the same ones (describe_camera_rejection) on every move
the pattern asks for, starting where a real run starts, at the Level-0 centre.
"""
import collections, os, sys, types, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ.setdefault('DRONE_CAMERA', 'hybrid')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flyby
from utils import describe_camera_rejection

FRAMES = int(os.environ.get('SIM_FRAMES', '249'))
state = flyby.Sequence()
here = (0, 1920, 1080)          # every run opens on the full frame
levels = collections.Counter()
refused = []
l2_centres = []

for frame in range(1, FRAMES + 1):
    levels[here[0]] += 1
    if here[0] == 2:
        l2_centres.append((here[1], here[2]))
    view = types.SimpleNamespace(resolution_level=here[0], center_x=here[1], center_y=here[2])
    request = types.SimpleNamespace(view=view, camera_command_feedback=None, frame=frame)
    command = flyby.choose_next_view(request, state)
    if command is None:
        refused.append((frame, here, None, 'no command issued'))
        continue
    target = (command.resolution_level, command.center_x, command.center_y)
    why = describe_camera_rejection(here[0], (here[1], here[2]), target[0], (target[1], target[2]))
    if why:
        refused.append((frame, here, target, why))
    else:
        here = target               # the evaluator applies it for the next frame

print(f'camera={flyby.CAMERA}  frames={FRAMES}')
print('level occupancy:', {f'L{k}': v for k, v in sorted(levels.items())},
      f'-> {100*levels[2]/FRAMES:.0f}% of frames at native resolution')
print(f'illegal or missing commands: {len(refused)}')
for row in refused[:8]:
    print('   frame %s  from %s -> %s : %s' % row)

if l2_centres:
    xs = sorted({x for x, _ in l2_centres})
    ys = sorted({y for _, y in l2_centres})
    print(f'L2 visited {len(set(l2_centres))} distinct centres; x {xs[0]}..{xs[-1]}, rows y={ys}')
    # An L2 view covers 960x540 of source, so count the top band actually swept.
    covered = set()
    for x, y in l2_centres:
        covered.update(range(max(0, x - 480) // 96, min(3840, x + 480) // 96))
    print(f'top-band horizontal coverage: {100*len(covered)/40:.0f}% of frame width')
