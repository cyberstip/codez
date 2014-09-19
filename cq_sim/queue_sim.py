from operator import itemgetter
import numpy as np
from scipy.stats import norm
import emcee


def sort_and_get_indices(arr):
  indices, sorted_arr = zip(*sorted(enumerate(arr), key=lambda x: x[1]))
  return list(sorted_arr), list(indices)


def same_sort(arr, indices):
  return [arr[i] for i in indices]


def unsort(sorted_arr, indices):
  return list(zip(*sorted(zip(indices, sorted_arr)))[1])


def convert_a_s_to_d(a, s, k):  # Used for initial estimate.
  ss = len(a) * [s]
  return convert_a_multi_s_to_d(a, ss, k)


def convert_a_multi_s_to_d(a, s, k):  # Used for initial estimate.
  finish_times = [None] * k
  actual_start = []
  actual_finishes = []
  sorted_a, indices = sort_and_get_indices(a)
  sorted_s = same_sort(s, indices)
  for aa, ss in zip(sorted_a, sorted_s):
    try:
      next_bot = finish_times.index(None)
      actual_start.append(aa)
    except ValueError:
      # Find bot that finishes first.
      next_bot = min(enumerate(finish_times), key=itemgetter(1))[0]
      actual_start.append(max(aa, finish_times[next_bot]))
    finish_times[next_bot] = actual_start[-1] + ss
    actual_finishes.append(finish_times[next_bot])

  return unsort(actual_finishes, indices)


def convert_a_d_to_u(a, d, k):
  """Given arrival and departure times, determine actual start time."""
  finish_times = [None] * k
  actual_start = []
  sorted_a, indices = sort_and_get_indices(a)
  sorted_d = same_sort(d, indices)
  for a, d in zip(sorted_a, sorted_d):
    try:
      next_bot = finish_times.index(None)
      actual_start.append(a)
    except ValueError:
      # Find bot that finishes first.
      next_bot = min(enumerate(finish_times), key=itemgetter(1))[0]
      actual_start.append(max(a, finish_times[next_bot]))
    finish_times[next_bot] = d

  return unsort(actual_start, indices)


def actuals_to_deltas(actuals, start=0):
  return [j - i for i, j in zip(np.concatenate(([start], actuals)), actuals)]


def convert_a_d_to_delta_s(a, d, k):
  actual_starts = convert_a_d_to_u(a, d, k)
  arrival_deltas = actuals_to_deltas(a)
  service_times = [d - u for d, u in zip(d, actual_starts)]

  return arrival_deltas, service_times


def exponential_likelihood(x, lam=1.0):
  if x < 0.0:
    return -np.inf
  else:
    return np.log(lam * np.exp(-(lam*x)))


def normal_likelihood(x, mean=1.0, stddev=1.0):
  if x < 0.0:
    return -np.inf
  else:
    return np.log(norm.pdf(x, loc=mean, scale=stddev))


def lnlike(theta, arrival_density, service_density, k):
  """Likelihood of the data given theta."""
  a, d = np.reshape(theta, (2, -1))
  
  if any(a < 0.0):
    return -np.inf

  if any(aa > dd for aa, dd in zip(a, d)):
    return -np.inf

  arrival_deltas, service_times = convert_a_d_to_delta_s(a, d, k)
  arrival_likelihood = sum(exponential_likelihood(
    delta, lam=arrival_density) for delta in arrival_deltas)
  service_likelihood = sum(normal_likelihood(
    service, mean=service_density, stddev=659.0) for service in service_times)

  return arrival_likelihood + service_likelihood
