import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
sns.set(style="darkgrid", palette="Set2")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', default='fool_json.json',
      help='The file to read results from')
  parser.add_argument('--percentile', default=90, type=int,
      help='what percentile to graph')
  parser.add_argument('--filter-seconds', default=200, type=int,
      help='filter out anything faster than this many seconds')
  args = parser.parse_args()

  with open(args.filename) as f:
    data = json.load(f)

  flattened_data = list(
      filter(
        lambda x: x > args.filter_seconds,
        itertools.chain.from_iterable(x[1] for x in data)
        ))

  def distplot():
    sns.distplot(numpy.array(flattened_data))

  def violinplot():
    unflattened_data = numpy.array([x[1] for x in data])
    sns.violinplot(unflattened_data)

  def tspercentile(percentile, filter_seconds):
    return [numpy.percentile(
      filter(lambda x: x > filter_seconds, x[1]), percentile) for x in data]

  def tsplot():
    sns.tsplot(tspercentile(args.percentile, args.filter_seconds))

  def loadplot():
    times = numpy.array([x[0] for x in data])
    sns.distplot(numpy.array(times))

  loadplot()

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
  ax = plt.gca()
  ax.xaxis.set_ticks([i * 24.0 for i in range(8)])
#  plt.grid()
  plt.show()

if __name__ == '__main__':
  main()
