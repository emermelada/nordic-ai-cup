import subprocess, sys
# Install the RL/EDA stack into the active venv. Mac (MPS): no extra torch index.
pkgs = ["gymnasium", "stable-baselines3", "pandas", "matplotlib", "scikit-learn"]
cmd = [sys.executable, "-m", "pip", "install"] + pkgs
r = subprocess.run(cmd, capture_output=True, text=True)
print("RC", r.returncode)
print(r.stdout[-3000:])
print(r.stderr[-2000:])