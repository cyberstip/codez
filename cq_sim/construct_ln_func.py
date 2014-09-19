import json
import numpy as np

import ln_func_models


def model_loader(filename):
  with open(filename) as f:
    model = json.load(f)
  
  return ln_func_models.Pipeline(model, initial=True)


def get_starter_point_arrays(model, duration, num_arrays):
  point_arrays = []
  initial_points = model.starter_points_source(duration)
  if np.isneginf(model.lnlike(initial_points)):
    print 'oh snap %s' % initial_points
  assert not np.isneginf(model.lnlike(initial_points))
  for _ in range(num_arrays):
    new_points = [x + np.random.rand() for x in initial_points]
    point_arrays.append(new_points)
  return point_arrays
