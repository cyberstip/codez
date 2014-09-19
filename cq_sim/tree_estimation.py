import contextlib
import datetime
import json
import numpy
import urllib

status_app = 'https://chromium-status.appspot.com'
status_url = '%s/allstatus?format=json&limit=1000' % status_app

with contextlib.closing(urllib.urlopen(status_url)) as f:
  messages = json.load(f)

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
seconds_per = total_seconds / float(len(time_duration))

dudes = [y for x, y in time_duration]
log_dudes = [numpy.log(x) for x in dudes]

exp = numpy.polyfit(log_dudes, dudes, 0)[0]

print seconds_per, exp
