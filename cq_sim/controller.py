import models
import tree_estimation


TREES = [
  'https://chromium-status.appspot.com',
  'https://blink-status.appspot.com',
]


def write_tree_estimates(status_app):
  per_second, exp = tree_estimation.calculate_tree_data(status_app)

  models.TreeStatistics(
      status_app=status_app,
      closures_per_second=per_second,
      closure_length_exponent=exp,
  ).put()


def write_trees():
  for status_app in TREES:
    write_tree_estimates(status_app)
