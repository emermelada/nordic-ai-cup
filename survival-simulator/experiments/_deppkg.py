import scipy, numpy, pygame, shapely, pydantic
print("scipy", scipy.__version__, "numpy", numpy.__version__)
try:
    import cma; print("cma OK")
except Exception as e:
    print("no cma:", e)