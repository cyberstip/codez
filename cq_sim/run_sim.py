import numpy as np
import emcee

import json
import construct_ln_func
import queue_sim
import sys


sample_time = 604800.0
#ntemps = 5
burn_in = 10
runs = 10

pipeline = construct_ln_func.model_loader('cool.json')
ndim = pipeline.expected_points_source(sample_time)[0]
#nwalkers = 5 * ndim
nwalkers = 50

print ndim

#p0 = []
#for _ in range(ntemps):
#  p0.append(
p0 =  construct_ln_func.get_starter_point_arrays(
        pipeline, sample_time, nwalkers)

def logp(x):
  return 0.0 # flat prior

def lnlike(theta, pipe):
  kwarg = pipe.lnlike(theta)
  return kwarg

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
    threads=1,
    args=[
      pipeline,
    ],
    live_dangerously=True,
)

pos, prob, state = sampler.run_mcmc(p0, burn_in)
sampler.reset()
sampler.run_mcmc(pos, runs)

res = []
for sample in sampler.flatchain:
  a, d = pipeline.get_a_d(sample)
  total_times = [dd - aa for dd, aa in zip(d, a)]
  res.append(zip(a, total_times))

with open('results.json', 'w') as f:
  json.dump(res, f, indent=2)
#  print '----'
#  print 'like %f' % pipeline.lnlike(sample[0])
#  print 'a:', list(a)
#  print 'd:', list(d)
#  print 'total_times', total_times
#  print 'qqqq'

total_wait = []
for sample in sampler.flatchain:
  arrival, departure = pipeline.get_a_d(sample)
  for a, d in zip(arrival, departure):
    total_wait.append(d - a)

print 'mean wait: %fs, 90p: %fs, 95p: %fs, 99p: %fs, max: %fs' % (
    np.mean(total_wait),
    np.percentile(total_wait, 90),
    np.percentile(total_wait, 95),
    np.percentile(total_wait, 99),
    np.max(total_wait))
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
