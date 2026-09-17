import torch, numpy, sys
print("py", sys.version.split()[0])
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
print("numpy", numpy.__version__)
import gymnasium
print("gym", gymnasium.__version__)
import stable_baselines3 as sb3
print("sb3", sb3.__version__)
