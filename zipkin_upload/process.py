import argparse
from contextlib import closing
from datetime import datetime
import functools
import json
import requests
import urllib2
import uuid
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


def builder_id(master, builder):
  return '%s/%s' % (master, builder)


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
    # self.rietveld_timestamp, useless?
    return {
        'ss': self.timestamp,
        'cr': self.timestamp,
        'host': self.job_dict['slave'],
        'service_name': builder_id(
            self.job_dict['master'], self.job_dict['builder']),
        }


class VerifierJobsUpdate(CQApiRecord):
  def __init__(self, *args, **kwargs):
    super(VerifierJobsUpdate, self).__init__(*args, **kwargs)

    self.job_updates = set()
    for master in self._fields['jobs'].itervalues():
      for builder in master.itervalues():
        for job in builder['rietveld_results']:
          if job['result'] != -1 and job['buildnumber'] is not None:
            self.job_updates.add(JobUpdate(
                self.timestamp,
                job,
                datetime.strptime(
                    builder['timestamp'], "%Y-%m-%d %H:%M:%S.%f")))


class VerifierStart(CQApiRecord):
  def __init__(self, *args, **kwargs):
    super(VerifierStart, self).__init__(*args, **kwargs)
    self.tryjobs = set()
    for mastername, master_builders in self._fields['tryjobs'].iteritems():
      for builder in master_builders.keys():
        self.tryjobs.add(builder_id(mastername, builder))


class VerifierRetry(CQApiRecord):
  pass


def return_new_jobs(jobs_updates):
  total_jobs = set()
  for update in jobs_updates:
    if isinstance(update, VerifierJobsUpdate):
      diff = update.job_updates - total_jobs
      if diff:
        total_jobs |= diff
        update.job_updates = diff
        yield update
    else:
      yield update


def chunk_attempts(records):
  current_chunk = None
  for record in records:
    if isinstance(record, PatchStart):
      current_chunk = [record]
    elif isinstance(record, PatchStop):
      if current_chunk is not None:
        current_chunk.append(record)
        yield current_chunk
        current_chunk = None
    elif current_chunk is not None:
      current_chunk.append(record)


def translate_retry_to_start(records):
  last_update = None
  for record in records:
    if isinstance(record, VerifierRetry):
      assert last_update is not None
      bad_jobs = {}
      for mastername, master in last_update._fields['jobs'].iteritems():
        for buildername, builder in master.iteritems():
          for job in builder['rietveld_results']:
            if job['result'] != 0:
              bad_jobs.setdefault(mastername, {})[buildername] = {}
      yield VerifierStart(
          timestamp=datetime_to_timestamp(record.timestamp),
          tags=record._tags,
          key=record._key,
          fields={'tryjobs': bad_jobs})
    elif isinstance(record, VerifierJobsUpdate):
      last_update = record
      yield record
    else:
      yield record


def link_jobs_to_trigger(records):
  triggers = {}
  for record in records:
    if isinstance(record, VerifierStart):
      for trigger in record.tryjobs:
        triggers[trigger] = record.timestamp
    elif isinstance(record, VerifierJobsUpdate):
      for job_update in record.job_updates:
        zk_data = job_update.zipkin_data()
        zk_data['cs'] = triggers[
            builder_id(job_update.job_dict['master'],
                       job_update.job_dict['builder'])]
        zk_data['sr'] = triggers[
            builder_id(job_update.job_dict['master'],
                       job_update.job_dict['builder'])]
        yield zk_data
    else:
      yield record


class PatchStart(CQApiRecord):
  pass


class PatchStop(CQApiRecord):
  pass



RECORD_TYPES = {
    'verifier_jobs_update': VerifierJobsUpdate,
    'verifier_start': VerifierStart,
    'verifier_retry': VerifierRetry,
    'patch_start': PatchStart,
    'patch_stop': PatchStop,
}


def constructor(record):
  return RECORD_TYPES.get(
      record['fields'].get('action'), lambda **kwargs: None)(**record)


def gen_zipkin_uuid():
  return uuid.uuid4().hex[0:15]


def datetime_to_timestamp(dt):
  return (dt - datetime(1970, 1, 1)).total_seconds()


class ZipkinSpan(object):
  def __init__(
      self, name, trace_id, span_id, parent_id, parent_name, annotations):
    self.name = name
    self.trace_id = trace_id
    self.span_id = span_id
    self.parent_id = parent_id
    self.annotations = []

    for annot_name, annot_value in annotations.iteritems():
      if annot_name in ('cs', 'cr'):
        self.annotations.append({
            'key': annot_name,
            'type': 'timestamp',
            'value': int(datetime_to_timestamp(annot_value) * 1000000),
            'name': parent_name,
            'host': {
                'ipv4': '10.10.10.10',
                'port': 1234,
                'service_name': parent_name,
            }
        })
        # timestamps are integers in microseconds
      elif annot_name in ('ss', 'sr'):
        self.annotations.append({
            'key': annot_name,
            'type': 'timestamp',
            'value': int(datetime_to_timestamp(annot_value) * 1000000),
            'name': self.name,
            'host': {
                'ipv4': '10.10.10.10',
                'port': 1234,
                'service_name': self.name,
            }
        })

  def render(self):
    result = {
        'trace_id': self.trace_id,
        'span_id': self.span_id,
        'name': self.name,
        'annotations': self.annotations,
    }

    if self.parent_id:
      result['parent_span_id'] = self.parent_id

    return result


def construct_spans(records):
  records = list(records)
  trace_id = gen_zipkin_uuid()

  root_span = ZipkinSpan('cq_service', trace_id, trace_id, None, 'cq_request', {
      'sr': records[0].timestamp,
      'ss': records[-1].timestamp,
      'cs': records[0].timestamp,
      'cr': records[-1].timestamp,
  })
  spans = [root_span]
  for record in records[1:-1]:
    spans.append(ZipkinSpan(
        record['service_name'],
        trace_id, gen_zipkin_uuid(), trace_id, 'cq_service', record))

  return spans


def send_to_restkin(url, spans):
  headers = {'content-type': 'application/json'}
  with open('bruh.json', 'w') as f:
    json.dump(spans, f, indent=2)
  r = requests.post(url, data=json.dumps(spans), headers=headers)
  print r


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('issue', type=int, help='rietveld issue')
  parser.add_argument(
      'patchset', type=int, nargs='?', help='patchset', default=1)
  parser.add_argument('--restkin-url',
                      default='http://localhost:6956/v1.0/restkin/trace',
                      help='URL to send restkin data')
  args = parser.parse_args()

  url_template = ('https://chromium-cq-status.appspot.com/'
                  'query/issue=%d/patchset=%d')

  url = url_template % (args.issue, args.patchset)

  with closing(urllib2.urlopen(url)) as f:
    items = json.load(f)

  sorted_records = sorted(filter(bool, map(constructor, items['results'])))
  print sorted_records[0].timestamp
  print datetime_to_timestamp(sorted_records[0].timestamp)
  print sorted_records[-1].timestamp
  print datetime_to_timestamp(sorted_records[-1].timestamp)
  spans = []
  for record_set in list(chunk_attempts(sorted_records))[0:1]:
    for span in construct_spans(
        link_jobs_to_trigger(translate_retry_to_start(
            return_new_jobs(record_set)))):
      spans.append(span.render())
  send_to_restkin(args.restkin_url, spans)

  return 0

if __name__ == '__main__':
  sys.exit(main())
