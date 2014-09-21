import cq_estimates
import datetime
import models
import tree_estimation


TREES = [
  'https://blink-status.appspot.com',
  'https://chromium-status.appspot.com',
]


PROJECTS = [
    'blink',
    'chromium',
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


def write_cq_project_load(project, segment_length=30, periodicity='weekly'):
  segment_data = cq_estimates.crawl_segments(
      project, segment_length=30, periodicity=periodicity)
  segments = []
  for seg, data in segment_data.iteritems():
    segments.append(models.CQLoadSegment(
      segment=seg,
      request_count=data['reqs'],
      segment_count=data['segs'],
      requests_per_second=data['rps'],
    ))

  models.CQRequestLoad(
      project=project,
      segment_length_minutes=segment_length,
      periodicity=periodicity,
      segments=segments,
  ).put()


def write_cq_load():
  for project in PROJECTS:
    write_cq_project_load(project)
