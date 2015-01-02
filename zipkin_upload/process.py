import argparse
from contextlib import closing
from datetime import datetime
import functools
import json
import urllib2
import sys

@functools.total_ordering
class CQApiRecord(object):
  def __init__(self, timestamp=None, tags=None, key=None, fields=None):
    self.timestamp = datetime.utcfromtimestamp(timestamp)
    self._tags = tags
    self._key = key
    self._fields = fields

    self.project = None
    self.issue = None
    self.patchset = None
    self.attempt = None

  def set_identifiers(self, project, issue, patchset, attempt):
    self.project = project
    self.issue = issue
    self.patchset = patchset
    self.attempt = attempt

  def __eq__(self, other):
    return self.timestamp == other.timestamp and self._tags == other._tags

  def __lt__(self, other):
    return self.timestamp < other.timestamp


class JobUpdate(object):
  def __init__(self, timestamp, job_dict, rietveld_timestamp):
    self.timestamp = timestamp
    self.job_dict = job_dict
    self.rietveld_timestamp = rietveld_timestamp

  def __eq__(self, other):
    return self.job_dict == other.job_dict

  def __hash__(self):
    return hash(json.dumps(self.job_dict, sort_keys=True))

  def __str__(self):
    return str(self.job_dict)

  def __repr__(self):
    return 'JobUpdate: %s' % repr(self.job_dict)

  def zipkin_data(self):
    return {
        'sr': self.job_dict['timestamp'],
        'ss': self.rietveld_timestamp,
        'cr': self.timestamp,
        'host': self.job_dict['slave'],
        'service_name': (self.job_dict['master'] + '/' +
          self.job_dict['builder']),
    }


class VerifierJobsUpdate(CQApiRecord):
  def __init__(self, *args, **kwargs):
    super(VerifierJobsUpdate, self).__init__(*args, **kwargs)

    self.job_updates = set()
    for master in self._fields['jobs'].itervalues():
      for builder in master.itervalues():
        for job in builder['rietveld_results']:
          if job['result'] != -1:
            self.job_updates.add(JobUpdate(self.timestamp, job,
              builder['timestamp']))


def return_new_jobs(jobs_updates):
  total_jobs = set()
  for update in jobs_updates:
    diff = update.job_updates - total_jobs
    if diff:
      total_jobs |= diff
      yield diff


def enumerate_attempt(records):
  attempt = None
  for record in records:
    if record._fields.get('action') == 'patch_start':
      if attempt is None:
        attempt = 0
      else:
        attempt = attempt + 1
    yield (attempt, record)


def constructor(record):
  if record['fields'].get('action') == 'verifier_jobs_update':
    return VerifierJobsUpdate(**record)
  else:
    return CQApiRecord(**record)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('issue', type=int, help='rietveld issue')
  parser.add_argument(
      'patchset', type=int, nargs='?', help='patchset', default=1)
  args = parser.parse_args()

  url_template = ('https://chromium-cq-status.appspot.com/'
                  'query/issue=%d/patchset=%d')

  url = url_template % (args.issue, args.patchset)

  with closing(urllib2.urlopen(url)) as f:
    items = json.load(f)

  sorted_records = sorted(map(constructor, items['results']))
  filtered_records = [x for x in sorted_records if 'action' in x._fields]

  jobs_updates = []
  for attempt, record in enumerate_attempt(filtered_records):
    if attempt != 0:
      continue
    if isinstance(record, VerifierJobsUpdate):
      jobs_updates.append(record)
    #item._tags
    #print item['tags'], 
    #print item['fields'].get('action'), item['tags']

  for new_jobs in return_new_jobs(jobs_updates):
    for job in new_jobs:
      print job.zipkin_data()

  return 0

if __name__ == '__main__':
  sys.exit(main())
