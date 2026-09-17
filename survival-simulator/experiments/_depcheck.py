import sys
print(sys.executable)
try:
    import numpy, scipy, pygame, shapely, pydantic
    print("all deps OK", numpy.__version__)
except Exception as e:
    print("MISSING:", e)