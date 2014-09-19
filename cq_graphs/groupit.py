import argparse
from datetime import datetime
from datetime import timedelta
from itertools import chain, groupby
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import json
sns.set(style="darkgrid", palette="Set2")


def earliest_sunday(utctimestamp):
  date = datetime.utcfromtimestamp(utctimestamp)
  weekday = (date.weekday() + 1) % 7
  date -= timedelta(days=weekday)
  return date.replace(hour=0, minute=0, second=0, microsecond=0)


def time_to_weektime(utctimestamp):
    return (datetime.utcfromtimestamp(utctimestamp) -
        earliest_sunday(utctimestamp)).total_seconds()

def weektime_to_chunk(weektime, chunking):
    return int(weektime) / chunking

def chunk_data(data, chunking):
  def keyfunc(v):
    return int(time_to_weektime(v[0])) / chunking
  return [[float(k),
    [[time_to_weektime(g[0]), g[1]] for g in group]] for k, group in groupby(
      sorted(data, key=keyfunc), keyfunc)]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', default='cool_json.json',
      help='The file to read results from')
  parser.add_argument('--output-filename', default='fool_json.json',
      help='The file to write results to')
  parser.add_argument('--chunking', default=3600,
      help='seconds to chunk the data')
  parser.add_argument('--output_times', action='store_true',
      help='seconds to chunk the data')
  parser.add_argument('--max_seconds',
      default=10000,
      type=int,
      help='discard runs over this long')
  parser.add_argument('--bandwidth',
      type=float,
      help='kernel bandwidth')
  args = parser.parse_args()

  with open(args.filename) as f:
    data = json.load(f)

  chunked_data = chunk_data(chain.from_iterable(data), args.chunking)
  with open(args.output_filename, 'w') as f:
   json.dump(
       chunked_data,
       f, indent=2)

  raw_flattened_data = list(
        chain.from_iterable([q[1] for q in x[1]
          if q[1] <= args.max_seconds]
          for x in chunked_data))

  raw_flattened_times = list(
        chain.from_iterable([q[0] / 3600.0
          for q in x[1]
          if q[1] <= args.max_seconds]
          for x in chunked_data))

  sorted_times = sorted(
      zip(raw_flattened_times, raw_flattened_data), key=lambda x: x[0])

  flattened_times, flattened_data = zip(*sorted_times)
  flattened_times = list(flattened_times)
  flattened_data = list(flattened_data)

  assert sorted(flattened_times) == flattened_times

  def distplot():
    sns.distplot(numpy.array(flattened_data))

  def violinplot():
    unflattened_data = numpy.array([[q[1] for q in x[1]] for x in chunked_data])
    sns.violinplot(unflattened_data)

  def tsvariance():
    return [numpy.var([g[1] for g in x[1]])
      for x in chunked_data]

  def tsplot():
    sns.tsplot(tsvariance())

  def loadplot():
    sns.distplot(numpy.array(flattened_times))

  def both_kde():
    marginal_kws = {}
    joint_kws = {}
    if args.bandwidth:
      marginal_kws['bw'] = args.bandwidth
      joint_kws['bw_x'] = args.bandwidth
      joint_kws['bw_y'] = args.bandwidth
    sns.jointplot(numpy.array(flattened_times),
        numpy.array(flattened_data), kind='kde',
        marginal_kws=marginal_kws,
        joint_kws=joint_kws,
        ylim=(0, args.max_seconds),
        xlim=(0, 167),
        size=7)
    #sns.kdeplot(joined_data)
    #ax = plt.gca()
    #ax.xaxis.set_ticks([i * (24.0) for i in range(8)])

  #def var_kde():
    # crappy variance

  both_kde()

  #tsplot()


  print '50 %f / 75 %f / 90 %f / 99 %f /mean %f' % (
      numpy.percentile(flattened_data, 50),
      numpy.percentile(flattened_data, 75),
      numpy.percentile(flattened_data, 90),
      numpy.percentile(flattened_data, 99),
      numpy.mean(flattened_data),
  )

#  initial_x = 0.0
#  final_x = 167.0

#  plt.plot(x, [yy for yy in y])
#  plt.scatter([initial_x, final_x], [0.0, 0.0], s=0.0)
#  plt.grid()
  plt.show()

if __name__ == '__main__':
  main()
