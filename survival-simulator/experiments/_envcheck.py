import importlib, sys
print('python', sys.version.split()[0])
for m in ['numpy','pandas','torch','gymnasium','stable_baselines3','pygame']:
    try:
        mod = importlib.import_module(m)
        print('OK', m, getattr(mod,'__version__','?'))
    except Exception as e:
        print('FAIL', m, repr(e)[:140])
try:
    import torch
    print('mps', torch.backends.mps.is_available())
except Exception:
    pass