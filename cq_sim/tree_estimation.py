import contextlib
import datetime
import json
import numpy

from google.appengine.api import urlfetch

def calculate_tree_data(status_app):
  status_url = '%s/allstatus?format=json&limit=1000' % status_app

  result = urlfetch.fetch(status_url, deadline=60)
  if result.status_code != 200:
    raise ValueError('for url %s status was %d' % (
      status_url, result.status_code))

  messages = json.loads(result.content)

  def date_from_string(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

  closure_state = 'can_commit_freely'
  edge_messages = []
  time_duration = []
  for msg in reversed(messages):
    if edge_messages and msg[closure_state] == edge_messages[-1][closure_state]:
      continue
    if msg[closure_state]:
      if edge_messages:
        closure_start_time = date_from_string(edge_messages[-1]['date'])
        open_start_time = date_from_string(msg['date'])
        duration = (open_start_time - closure_start_time).total_seconds()
        time_duration.append((closure_start_time, duration))
    edge_messages.append(msg)

  total_seconds = (time_duration[-1][0] - time_duration[0][0]).total_seconds()
  per_second = float(len(time_duration)) / total_seconds

  durations = [y for x, y in time_duration]
  log_durations = [numpy.log(x) for x in durations]

  exp = numpy.polyfit(log_durations, durations, 0)[0]

  return per_second, exp
