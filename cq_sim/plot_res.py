import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
sns.set(style="darkgrid", palette="Set2")

with open('results.json') as f:
  data = json.load(f)

modulo = 3600

if not data:
  raise ValueError('no data!')

max = 0
for set in data:
  for entry in set:
    if entry[0] > max:
      max = entry[0]

traces = []
for set in data:
  traces.append(list([] for _ in range(int(max / modulo) + 1)))
  for entry in set:
    traces[-1][int(entry[0] / modulo)].append(entry[1])

new_traces = []
for trace in traces:
  new_traces.append(list(
      np.median(x) if x else float('nan') for x in trace))
sns.tsplot(new_traces)
plt.show()
