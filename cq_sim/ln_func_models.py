import copy
import numpy as np

import queue_sim

import scipy
import sys


class Source(object):
  type = 'source'

  def starter_points_source(self, duration):
    # return [1.0 / self.rate * i for i in range(points)]
    raise NotImplementedError

  def expected_points_source(self, duration):
    # return int(self.rate * duration), int(self.rate * duration)
    # added points / output size
    raise NotImplementedError

  def output(self, theta_arr):
    return theta_arr

  def get_a_d(self, theta_arr):
    return theta_arr, self.output(theta_arr)

  def lnlike(self, theta_arr):
    raise NotImplementedError


class Pipe(object):
  type = 'pipe'

  def expected_points(self, point_len, _duration):
    self.input_point_len = point_len
    self.point_allocation = [point_len]
    return point_len, point_len

  def starter_points(self, points, duration):
    raise NotImplementedError

  def _input_arr(self, theta_arr):
    return list(theta_arr)[0:self.input_point_len]

  def output_arr(self, theta_arr):
    return list(theta_arr)[-self.point_allocation[-1]:]

  def lnlike(self, theta_arr):
    raise NotImplementedError


class ConstantPoissonSource(Source):
  def __init__(self, params):
    super(ConstantPoissonSource, self).__init__()
    assert params.keys() == ['rate']
    assert params['rate'] > 0.0

    self.rate = params['rate']

  def starter_points_source(self, duration):
    points, _ = self.expected_points_source(duration)
    return [1.0 / self.rate * (i + 1) for i in range(points)]

  def expected_points_source(self, duration):
    return int(self.rate * duration), int(self.rate * duration)

  def lnlike(self, theta):
    # calculate arrival delta, theta will be in absolute times
    deltas = [x - y for x, y in zip(theta, [0] + theta[:-1])]
    return sum(queue_sim.exponential_likelihood(
      delta, lam=self.rate) for delta in deltas)


class PeriodicPoissonSource(Source):
  """Inelegant stair-step poisson."""
  def __init__(self, params):
    super(PeriodicPoissonSource, self).__init__()
    assert sorted(params.keys()) == ['periodicity', 'rates']
    assert params['periodicity'] > 0
    assert len(params['rates']) > 0
    assert all(rate > 0.0 for rate in params['rates'])
    assert params['periodicity'] % len(params['rates']) == 0

    self.rates = params['rates']
    self.periodicity = params['periodicity']
    self.span = params['periodicity'] / len(params['rates'])
    self.periodic_count = self.span * sum(self.rates)

  def _find_absolute_span(self, abstime):
    return int(abstime) / self.span

  def _find_span(self, abstime):
    return self._find_absolute_span(abstime) % len(self.rates)

  def _find_rate(self, abstime):
    return self.rates[self._find_span(abstime)]

  def _offset_from_span(self, abstime):
    abstime -= self.periodicity * (int(abstime) / self.periodicity)
    offset = abstime - (self.span * self._find_span(abstime))
    return offset

  def _integrate_between_two_points(self, timeone, timetwo):
    reverse = timetwo < timeone

    if reverse:
      timeone, timetwo = timetwo, timeone
    timeone_offset = self.span - self._offset_from_span(timeone)
    running_count = timeone_offset * self._find_rate(timeone)
    timeone += timeone_offset

    timetwo_offset = self.span - self._offset_from_span(timetwo)
    running_count -= timetwo_offset * self._find_rate(timetwo)  # Note minus
    timetwo += timetwo_offset

    full_zones = ((self._find_absolute_span(timetwo) -
      self._find_absolute_span(timeone)) / len(self.rates))

    running_count += full_zones * self.periodic_count
    timeone += full_zones * self.periodicity

    abs_timetwo = self._find_absolute_span(timetwo)
    while self._find_absolute_span(timeone) < abs_timetwo:
      running_count += self.span * self._find_rate(timeone)
      timeone += self.span

    if reverse:
      return -running_count
    return running_count

  def expected_points_source(self, duration):
    """Integrate over rates."""
    expected_points = int(self._integrate_between_two_points(0.0, duration))
    return expected_points, expected_points

  def _find_next_point(self, last_point):
    """Finds the expected next point in time."""
    def func_to_optimize(x):
      return self._integrate_between_two_points(last_point, x) - 1.0
    estimate = last_point + (self.periodicity / self.periodic_count)
    return scipy.optimize.newton(func_to_optimize, estimate)

  def starter_points_source(self, duration):
    """Integrate over rates."""
    starter_points = []
    last_point = 0.0
    for _ in range(self.expected_points_source(duration)[1]):
      last_point = self._find_next_point(last_point)
      starter_points.append(last_point)
    return starter_points

  def lnlike(self, theta):
    # First calculate arrival delta, theta will be in absolute times
    deltas = [x - y for x, y in zip(theta, [0] + theta[:-1])]
    avg_rates = [
        self._integrate_between_two_points(y, x) / ((x - y) or 1.0)
        for x, y in zip(theta, [0] + theta[:-1])]
    if any(delta <= 0.0 for delta in deltas):
      return -np.inf
    return sum(queue_sim.exponential_likelihood(
      delta, lam=avg_rate) for delta, avg_rate in zip(deltas, avg_rates))


class KServerPipeline(Pipe):
  def __init__(self, params):
    super(KServerPipeline, self).__init__()
    assert sorted(params.keys()) == ['pipeline', 'servers']
    assert params['servers'] > 0

    self.servers = params['servers']
    self.pipeline = Pipeline(params['pipeline'])

  def expected_points(self, point_len, _duration):
    # Can't use duration in a process pipeline.
    pipeline_pts = self.pipeline.expected_points(1, None)
    self.point_len = point_len
    self.pipeline_pts = pipeline_pts[0]
    return (point_len * pipeline_pts[0]), (point_len * pipeline_pts[1])

  def starter_points(self, points, _duration):
    starter_points = self.pipeline.starter_points([0.0], None)
    return starter_points * len(points)

  def _get_input_and_pipeline_thetas(self, theta):
    input = theta[:-self.point_len * pipeline_pts]
    pipeline_theta = theta[-self.point_len * pipeline_pts:]

    def chunks(l, n):
      n = max(1, n)
      return [l[i:i + n] for i in range(0, len(l), n)]
    return list(chunks(pipeline_theta, pipeline_pts))

  def output_arr(self, theta):
    input, pipeline_thetas = self._get_input_and_pipeline_thetas(theta)
    s = []
    for a, pipeline_theta in zip(input, pipeline_thetas):
      s.append(self.pipeline.get_a_d([a] + pipeline_theta)[1][0])

    self.pipeline.expected_points(len(points), duration)
    return queue_sim.convert_a_multi_s_to_d(
        points, s, self.servers)

  def lnlike(self, theta):
    input, pipeline_thetas = self._get_input_and_pipeline_thetas(theta)

    return sum(self.pipeline.lnlike([a] + pipeline_theta)
               for a, pipeline_theta in zip(input, pipeline_theta))


class GaussianDelay(Pipe):
  def __init__(self, params):
    super(GaussianDelay, self).__init__()
    assert sorted(params.keys()) == ['mean', 'stddev']
    assert params['mean'] > 0.0
    assert params['stddev'] > 0.0

    self.mean = params['mean']
    self.stddev = params['stddev']

  def starter_points(self, points, _duration):
    return [x + self.mean for x in points]

  def lnlike(self, theta):
    a, d = np.reshape(theta, (2, -1))

    if any(dd < 0.0 for dd in d):
      return -np.inf

    if any(aa > dd for aa, dd in zip(a, d)):
      return -np.inf

    service_times = [d - a for d, a in zip(d, a)]

    return sum(queue_sim.normal_likelihood(
      service, mean=self.mean, stddev=self.stddev) for service in service_times)


class ConstantTreeDelay(Pipe): 
  name = 'constant_tree_delay'

  def __init__(self, params):
    super(ConstantTreeDelay, self).__init__()
    assert sorted(params.keys()) == ['length', 'rate']
    assert params['length'] > 0.0
    assert params['rate'] > 0.0

    self.rate = params['rate']
    self.length = params['length']

  def expected_points(self, point_len, duration):
    self._number_of_closures = int(self.rate * duration)
    return 2 * self._number_of_closures, point_len

  @staticmethod
  def shift_points_to_tree_open(points, closures, closure_lengths):
    # assumes closures are sorted, like points are
    closure_idx = 0
    for point in points:
      if not closures:
        yield point

      # guarantee we seek to the highest closure lower or equal to us
      while ((closure_idx + 1) < len(closures) and
          point >= closures[closure_idx + 1]):
        closure_idx += 1
      closure_end = closures[closure_idx] + closure_lengths[closure_idx]
      if point <= closure_end and point >= closures[closure_idx]:
        yield closure_end
      else:
        yield point

  def starter_points(self, points, duration):
    self.expected_points(len(points), duration)
    closure_times = [
        1.0 / self.rate * (1 + i) for i in range(self._number_of_closures)]
    closure_lengths = [self.length for i in range(self._number_of_closures)]
    return closure_times + closure_lengths

  def lnlike(self, theta):
    lengths = theta[-self._number_of_closures:]
    closures = theta[-(2 * self._number_of_closures):-self._number_of_closures]
    deltas = [x - y for x, y in zip(closures, [0] + closures[:-1])]
    return sum(queue_sim.exponential_likelihood(
      delta, lam=self.rate) for delta in deltas) + sum(
          queue_sim.exponential_likelihood(
            length, lam=self.length) for length in lengths)

  def output_arr(self, theta_arr):
    points = theta_arr[:(-2 * self._number_of_closures)]
    lengths = theta_arr[-self._number_of_closures:]
    closures = theta_arr[
        -(2 * self._number_of_closures):-self._number_of_closures]
    sorted_points, indices = queue_sim.sort_and_get_indices(points)
    unsorted_out = list(
        self.shift_points_to_tree_open(sorted_points, closures, lengths))
    return queue_sim.unsort(unsorted_out, indices)


class Duplicator(Pipe):
  name = 'duplicator'
  def __init__(self, params):
    super(Duplicator, self).__init__()
    assert sorted(params.keys()) == ['archetype', 'map_params']
    assert isinstance(params['archetype'], dict)
    assert isinstance(params['map_params'], list)

    map_keys = []
    maps = []
    keys_length = None
    for k, v in params['map_params'].iteritems():
      if keys_length is None:
        keys_length = len(v)
      else:
        assert len(v) == keys_length
      map_keys.append(k)
      maps.append(v)

    self.pipes = []
    for param_set in zip(*maps):
      self.pipes.append(
          Pipeline(
            self.apply_params(
              params['archetype'], 
              map_keys,
              param_set)))

  def recursive_apply(self, obj, keys, params):
    for k, v in zip(keys, params):
      if isinstance(obj, list):
        for idx, item in enumerate(obj):
          if item == k:
            obj[idx] = v
          elif isinstance(item, list) or isinstance(item, dict):
            self.recursive_apply(item, keys, params)
      elif isinstance(obj, dict):
        for ok, ov in obj.copy().iteritems():
          if ov == k:
            obj[ok] = v
          elif isinstance(ov, list) or isinstance(ov, dict):
            self.recursive_apply(ov, keys, params)

  def apply_params(self, archetype, keys, params):
    new_archetype = copy.deepcopy(archetype)
    self.recursive_apply(new_archetype['params'], keys, params)
    return new_archetype

  def expected_points(self, point_len, duration):
    self.input_point_len = point_len
    self.point_allocation = []
    for pipe in self.pipes:
      self.point_allocation.extend(pipe.expected_points(point_len, duration))
    assert all(
        alloc == self.point_allocation[1]
        for alloc in self.point_allocation[1:])
    return (
        sum(x[0] for x in self.point_allocation),
        self.point_allocation[-1][1])

  def starter_points(self, points, duration):
    self.expected_points(len(points), duration)
    new_points = []
    for pipe in self.pipes:
      new_points.extend(pipe.starter_points(new_points, duration))
    return new_points

  def output(self, theta_arr):
    theta = list(theta_arr)
    running_count = self.input_arr(theta)
    max_local_theta = theta[running_count:running_count +
      self.point_allocation[0][0]]
    maximum = max(max_local_theta)
    for allocation in self.point_allocation:
      local_theta = theta[running_count:running_count + allocation[0]]
      running_count += allocation[0]
      if max(local_theta) > maximum:
        maximum = max(local_theta)
        max_local_theta = local_theta
    return max_local_theta
        
        
  def lnlike(self, theta):
    theta = list(theta)
    results = []
    running_count = self.input_arr(theta)
    for idx, allocation in enumerate(self.point_allocation):
      local_theta = self.input_arr(theta) + theta[
          running_count:running_count + allocation[0]]
      running_count += allocation[0]
      results.append(self.pipes[idx].lnlike(local_theta))
    return sum(results)


class Amalgamator(Source):
  def __init__(self, params):
    super(Amalgamator, self).__init__()
    assert sorted(params.keys()) == ['source_pipelines']
    assert len(params['source_pipelines']) > 0

    self.source_pipelines = [Pipeline(x, initial=True)
        for x in params['source_pipelines']]

  def starter_points_source(self, duration):
    points = []
    for pipe in self.source_pipelines:
      points.extend(pipe.starter_points_source(duration))
    return points

  def expected_points_source(self, duration):
    points = [pipe.expected_points_source(duration)
              for pipe in self.source_pipelines]
    self.point_allocation = list(zip(*points)[0])
    return sum(self.point_allocation), sum(zip(*points)[1])

  def _point_allocation_slices(self, theta):
    theta = list(theta)
    thetas = []
    running_count = 0
    for count in self.point_allocation:
      thetas.append(theta[running_count:running_count + count])
      running_count += count
    return thetas

  def _get_a_d_i_sort(self, theta):
    thetas = self._point_allocation_slices(theta)
    aa = []
    dd = []
    source_indices = []
    source_index = 0
    for pipe, mini_theta in zip(self.source_pipelines, thetas):
      a, d = pipe.get_a_d(mini_theta)
      aa.extend(a)
      dd.extend(d)
      source_indices.extend([source_index] * len(a))
      source_index += 1
    aaa, indices = queue_sim.sort_and_get_indices(aa)
    ddd = queue_sim.same_sort(dd, indices)
    iii = queue_sim.same_sort(source_indices, indices)
    return aaa, ddd, iii, indices

  def output(self, theta):
    _, d, _, sort = self._get_a_d_i_sort(theta)
    return queue_sim.unsort(d, sort)

  def lnlike(self, theta):
    thetas = self._point_allocation_slices(theta)
    lnlikes = []
    for pipe, mini_theta in zip(self.source_pipelines, thetas):
      lnlikes.append(pipe.lnlike(mini_theta))
    return sum(lnlikes)


class KServerAmalgamator(Source):
  def __init__(self, params):
    super(KServerAmalgamator, self).__init__()

    assert sorted(params.keys()) == [
        'process_pipelines', 'servers', 'source_pipelines']

    self.process_pipelines = [Pipeline(p) for p in params['process_pipelines']]
    self.servers = params['servers']
    self.source_pipelines = [Pipeline(x, initial=True)
        for x in params['source_pipelines']]

    assert(self.servers > 0)
    assert len(params['source_pipelines']) > 0
    assert len(self.process_pipelines) == len(self.source_pipelines)

  def lnlike(self, theta):
    source_thetas, process_thetas = self._get_pipe_process_thetas(theta)

    return sum(source.lnlike(source_theta) + process.lnlike(process_theta)
               for source, process in zip(source_thetas, process_thetas))

  def starter_points_source(self, duration):
    points = []
    for pipe, process in zip(self.source_pipelines, self.process_pipelines):
      pts = pipe.starter_points_source(duration)
      points.extend(pts)
      _, d = pipe.get_a_d(pts)
      points.extend(process.starter_points(d, duration))
    return points

  def _get_pipe_process_thetas(self, theta):
    source_pipe_thetas = []
    process_pipe_thetas = []

    param_size = sum(a + b for a, b in self.point_allocation)

    running_count = 0
    for source, process in self.point_allocation:
      source_pipe_thetas.append(theta[running_count:running_count + source])
      running_count += source
      process_pipe_thetas.append(theta[running_count:running_count + process])
      running_count += process

    assert (len(source_pipe_thetas) + len(process_pipe_thetas) == len(theta))

    return source_pipe_thetas, process_pipe_thetas

  def expected_points_source(self, duration):
    self.point_allocation = []
    points = []
    for pipe, process in zip(self.source_pipelines, self.process_pipelines):
      pts = pipe.expected_points_source(duration)
      expected = process.expected_points(pts[1], duration)
      self.point_allocation.append([pts[0], expected[0]])
      points.append((expected[0] + pts[0], expected[1]))

    print points
    res = sum(zip(*points)[0])
    out = sum(zip(*points)[1])
    return res, out

  def get_a_d(self, theta):
    source_pipe_thetas, process_pipe_thetas = self._get_pipe_process_thetas(
        theta)

    input_from_sources = []
    for group in [pipe.get_a_d(thetas)[0]
        for pipe, thetas in zip(
          self.source_source_pipelines, source_pipe_thetas)]:
        input_from_sources.extend(group)
    return input_from_sources, self.output(theta)

  def output(self, theta):
    source_pipe_thetas, process_pipe_thetas = self._get_pipe_process_thetas(
        theta)

    initial_points = [pipe.get_a_d(thetas)[1]
        for pipe, thetas in zip(
          self.source_source_pipelines, source_pipe_thetas)]

    processed_points = [process.get_a_d(input + thetas)[1]
        for input, process, thetas in zip(
          initial_points, self.process_pipelines, process_pipe_thetas)]

    delta_s = []
    for initial, processed in zip(initial_points, processed_points):
      delta_s.extend(d - a for a, d in zip(initial, processed))

    return queue_sim.convert_a_multi_s_to_d(
        initial_points, delta_s, self.servers)


class Pipeline(object):
  type = 'pipeline'

  def __init__(self, pipeline, initial=False):
    assert isinstance(pipeline, list)
    assert len(pipeline) > 0
    assert all(isinstance(e, dict) for e in pipeline)
    assert all((k in ['name', 'params'] for k in e.keys()) for e in pipeline)
    assert all(e['name'] in PIPELINE_MODELS for e in pipeline)
    assert all(isinstance(e.get('params', {}), dict) for e in pipeline)

    pipeline_objs = [
        PIPELINE_MODELS[e['name']](e.get('params', {})) for e in pipeline]

    if initial == True:
      self.source = pipeline_objs[0]
      assert self.source.type == 'source'
      self.pipes = pipeline_objs[1:]
    else:
      self.source = None
      self.pipes = pipeline_objs
    assert all(o.type == 'pipe' for o in self.pipes)

  def start_count(self, source=False):
    count = self.point_allocation[0][0]
    if not source:
      count -= self.point_allocation[0][1]
    return count 

  def final_output_count(self):
    return self.point_allocation[-1][1]

  def get_a_d(self, theta):
    _, last_out = self._execute(theta)
    a = theta[0:self.start_count(source=self.source)]

    assert len(a) == len(last_out)
    return (a, last_out)

  def starter_points(self, input_points, duration):
    self.expected_points(len(input_points), duration)
    points = self.pipes[0].starter_points(input_points, duration)
    for idx, obj in enumerate(self.pipes[1:]):
      last_output = self.point_allocation[idx][1]
      points.extend(obj.starter_points(points[-last_output:], duration))

    return points

  def starter_points_source(self, duration):
    self.expected_points_source(duration)
    points = self.source.starter_points_source(duration)
    last_obj = self.source
    for idx, obj in enumerate(self.pipes):
      last_output = self.point_allocation[idx + 1][1]
      points.extend(obj.starter_points(last_obj.output(points[-last_output:]), duration))
      last_obj = obj

    return points

  def expected_points(self, points, duration):
    self.point_allocation = [self.pipes[0].expected_points(points, duration)]
    self.point_allocation[0] = (
        self.point_allocation[0][0] + points, self.point_allocation[0][1])
    for obj in self.pipes[1:]:
      self.point_allocation.append(
          obj.expected_points(self.point_allocation[-1][1]))
    return (
        sum(x[0] for x in self.point_allocation),
        self.point_allocation[-1][1])

  def expected_points_source(self, duration):
    self.point_allocation = [self.source.expected_points_source(duration)]
    for obj in self.pipes:
      self.point_allocation.append(
          obj.expected_points(self.point_allocation[-1][1], duration))
    return (
        sum(x[0] for x in self.point_allocation),
        self.point_allocation[-1][1])

  def lnlike(self, theta_arr):
    lnlikes, _ = self._execute(theta_arr)
    return sum(lnlikes)

  def _execute(self, theta_arr):
    theta = list(theta_arr)
    thetas = []
    running_count = 0
    for count, output in self.point_allocation:
      thetas.append(theta[running_count:running_count + count])
      running_count += count

    if self.source:
      lnlikes = [self.source.lnlike(thetas[0])]
      last_out = list(self.source.output(thetas[0]))
      for idx, obj in enumerate(self.pipes):
        lnlikes.append(obj.lnlike(last_out + thetas[idx + 1]))
        last_out = obj.output_arr(last_out + thetas[idx + 1])
    else:
      lnlikes = []
      last_out = []
      for idx, obj in enumerate(self.pipes):
        lnlikes.append(obj.lnlike(last_out + thetas[idx]))
        last_out = obj.output_arr(last_out + thetas[idx])
    return lnlikes, last_out


PIPELINE_MODELS = {
    'constant_poisson_source': ConstantPoissonSource,
    'periodic_poisson_source': PeriodicPoissonSource,
    'k_server_pipeline': KServerPipeline,
    'gaussian_delay': GaussianDelay,
    'constant_tree_delay': ConstantTreeDelay,
    'duplicator': Duplicator,
    'amalgamator': Amalgamator,
    'k_server_amalgamator': KServerAmalgamator,
}
