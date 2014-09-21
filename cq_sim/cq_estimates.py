import contextlib
import collections
import datetime
import json
import logging
import urllib

from google.appengine.api import urlfetch

import time_helpers


def get_crawl_segments(
    start=None, end=None, segment_length=30, periodicity='weekly'):
  start = start or datetime.datetime.now()
  end = end or (start - datetime.timedelta(days=30))
  segments, seg_counts = time_helpers.generate_time_sequence(
      start,
      end,
      segment_length=segment_length,
      periodicity=periodicity,
  )

  return segments, seg_counts


def query_cq_status(project, action, params=None):
  url = 'https://chromium-cq-status.appspot.com/query/project=%s/action=%s' % (
      project, action)

  if params:
    url = url + '?' + urllib.urlencode(params)

  result = urlfetch.fetch(url)
  if result.status_code != 200:
    raise ValueError('for url %s status was %d' % (
      url, result.status_code))

  logging.info('crawling %s...' % url)

  return_json = json.loads(result.content)
  results = return_json['results']
  if return_json['more']:
    cursor_params = params.copy()
    cursor_params['cursor'] = return_json['cursor']
    results += query_cq_status(project, action, cursor_params)
  return results
  

# TODO(stip): Make async + parallel.
def crawl_segments(
    project, start=None, end=None, segment_length=30, periodicity='weekly'):
  segments, seg_counts = get_crawl_segments(
      start, end, segment_length=30, periodicity='weekly')

  timestamp_segments = [
      (time_helpers.to_timestamp(start), time_helpers.to_timestamp(end))
      for start, end in segments]


  counts = collections.defaultdict(lambda: {'reqs': 0, 'segs': 0, 'rps': 0.0})
  for start, end in timestamp_segments:
    json_data = query_cq_status(project, 'patch_start', params={
      'begin': start,
      'end': end,
    })

    seg, _ = time_helpers.bin_time_to_segment(
        datetime.datetime.utcfromtimestamp(start),
        segment_length=segment_length, periodicity=periodicity)

    counts[seg]['reqs'] += len(json_data)

  for seg, count in seg_counts.iteritems():
    counts[seg]['segs'] = count
    reqs = counts[seg]['reqs']
    total_secs = counts[seg]['segs'] * segment_length * 60
    counts[seg]['rps'] = float(reqs) / float(total_secs)

  return counts
