import calendar
import collections
import datetime


def weekly_start(time):
  start_day = time
  # Go back until we hit Sunday.
  while start_day.weekday():
    start_day = start_day - datetime.timedelta(days=1)

  # Start at the beginning on Sunday.
  return start_day.replace(hour=0, minute=0, second=0, microsecond=0)


PERIODICITIES = {
  'weekly': {
    'periodicity_length': 7 * 24 * 60,
    'start_func': weekly_start,
  }
}


def get_start_day(time, periodicity='weekly'):
  return PERIODICITIES[periodicity]['start_func'](time)


def bin_time_to_segment(time, segment_length=30, periodicity='weekly'):
  periodicity_length = PERIODICITIES[periodicity]['periodicity_length']
  segment_length_cleanly_divides_period = (
      periodicity_length % segment_length) == 0
  assert segment_length_cleanly_divides_period

  start_day = get_start_day(time, periodicity=periodicity)

  segment = int(((time - start_day).total_seconds() / 60)) / segment_length
  segment_start = start_day + datetime.timedelta(minutes=segment_length*segment)
  return segment, segment_start


def segment_cardinality(segment_length=30, periodicity='weekly'):
  return PERIODICITIES[periodicity]['periodicity_length'] / segment_length


def generate_time_sequence(start, end, segment_length=30, periodicity='weekly'):
  assert start >= end
  start_seg, start_time = bin_time_to_segment(
      start,
      segment_length=segment_length,
      periodicity=periodicity)

  cardinality = segment_cardinality(
      segment_length=segment_length, periodicity=periodicity)
  segments = []
  seg_time = start_time - datetime.timedelta(minutes=segment_length)
  seg = (start_seg - 1) % cardinality

  seg_counts = collections.defaultdict(int)
  while seg_time >= end:
    segments.append(
        (seg_time, seg_time + datetime.timedelta(minutes=segment_length)))
    seg_counts[seg] += 1

    seg_time = seg_time - datetime.timedelta(minutes=segment_length)
    seg = (seg - 1) % cardinality

  return segments, seg_counts


def to_timestamp(date):
  return calendar.timegm(date.utctimetuple())
