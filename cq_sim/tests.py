import unittest
from hypothesis import given, assume
import hypothesis.strategies as st
import numpy as np

import queue_sim
import ln_func_models


EPSILON = 0.001


class StatisticalTestQueueSim(unittest.TestCase):
  @given(st.lists(st.floats()))
  def testSorted(self, ints):
    sorty, indices = queue_sim.sort_and_get_indices(ints)
    self.assertEqual(sorty, sorted(sorty))
    self.assertEqual(sorty, queue_sim.same_sort(ints, indices))

  @given(st.lists(st.floats()))
  def testUnsort(self, ints):
    sorty, indices = queue_sim.sort_and_get_indices(ints)
    self.assertEqual(ints, queue_sim.unsort(sorty, indices))

  @given(
      st.lists(st.floats()),
      st.floats(min_value=0.0),
      st.integers(min_value=1))
  def testConvertASToD(self, a, s, k):
    assume(all(np.isfinite(aa) for aa in a))
    assume(np.isfinite(s))

    d = queue_sim.convert_a_s_to_d(a, s, k)
    self.assertTrue(all(dd >= (s + aa) for aa, dd in zip(a, d)))

  @given(
      st.lists(st.floats()),
      st.lists(st.floats(min_value=0.0)),
      st.integers(min_value=1))
  def testConvertMultiASToD(self, a, s, k):
    assume(len(a) == len(s))
    assume(all(np.isfinite(aa) for aa in a))
    assume(all(np.isfinite(ss) for ss in s))

    d = queue_sim.convert_a_multi_s_to_d(a, s, k)
    self.assertTrue(all(dd >= (ss + aa) for aa, ss, dd in zip(a, s, d)))

  @given(
      st.lists(st.floats()),
      st.lists(st.floats(min_value=0.0)),
      st.integers(min_value=1))
  def testConvertADToU(self, a, q, k):
    assume(len(a) == len(q))
    assume(all(np.isfinite(aa) for aa in a))
    assume(all(np.isfinite(qq) for qq in q))
    d = [aa + qq for aa, qq in zip(a, q)]

    u = queue_sim.convert_a_d_to_u(a, d, k)
    self.assertTrue(all(uu >= aa for aa, uu in zip(a, u)))
    self.assertTrue(all(uu <= dd for uu, dd in zip(u, d)))

  @given(
      st.lists(st.floats()),
      st.lists(st.floats(min_value=0.0)),
      st.integers(min_value=1))
  def testConvertADToDeltaS(self, a, q, k):
    assume(len(a) == len(q))
    assume(all(np.isfinite(aa) for aa in a))
    assume(all(np.isfinite(qq) for qq in q))
    d = [aa + qq for aa, qq in zip(a, q)]

    _, s = queue_sim.convert_a_d_to_delta_s(a, d, k)
    self.assertTrue(all((ss + aa) <= dd for aa, dd, ss in zip(a, d, s)))

class StatisticalTestLnFuncModels(unittest.TestCase):
  @given(
      st.floats(min_value=EPSILON),
      st.floats(min_value=EPSILON),
      st.lists(st.floats(min_value=EPSILON), min_size=1))
  def testConstantPoissonSource(self, rate, duration, deltas):
    assume(np.isfinite(rate))
    assume(np.isfinite(duration))
    assume(all(np.isfinite(d) for d in deltas))
    cps = ln_func_models.ConstantPoissonSource(params={'rate': rate})

    expected_point_count = cps.expected_points_source(duration)[0]
    self.assertGreaterEqual(expected_point_count, 0.0)

    starter_points = cps.starter_points_source(duration)
    self.assertEqual(len(starter_points), expected_point_count)

    theta = [deltas[0]]
    for d in deltas[1:]:
      theta.append(theta[-1] + d)

    res = cps.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  @given(
      st.lists(st.floats(min_value=EPSILON), min_size=1),
      st.floats(min_value=EPSILON),
      st.lists(st.floats(min_value=EPSILON), min_size=1))
  def testPeriodicPoissonSource(self, rates, duration, deltas):
    assume(all(np.isfinite(r) for r in rates))
    assume(np.isfinite(duration))
    assume(all(np.isfinite(d) for d in deltas))
    periodicity = len(rates)
    pps = ln_func_models.PeriodicPoissonSource(
        params={'rates': rates, 'periodicity': periodicity})

    expected_point_count = pps.expected_points_source(duration)[0]
    self.assertGreaterEqual(expected_point_count, 0.0)

    starter_points = pps.starter_points_source(duration)
    self.assertEqual(len(starter_points), expected_point_count)

    theta = [deltas[0]]
    for d in deltas[1:]:
      theta.append(theta[-1] + d)
    res = pps.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  @given(
      st.floats(min_value=EPSILON),
      st.floats(min_value=EPSILON),
      st.lists(st.floats(min_value=EPSILON), min_size=1),
      st.integers(min_value=1))
  def testKServerPipeline(self, rate, duration, deltas, k):
    assume(np.isfinite(rate))
    assume(np.isfinite(duration))
    assume(all(np.isfinite(d) for d in deltas))
    assume(np.isfinite(duration))

    ksp = ln_func_models.Pipeline([{
      'name': 'constant_poisson_source',
      'params': {'rate': rate}
    },
    {
      'name': 'k_server_pipeline',
      'params': {
        'servers': k,
        'pipeline': [{
        'name': 'gaussian_delay',
        'params': {'mean': 10.0,
                   'stddev': 1.0}
      }],
      }
    }],
    initial=True)

    expected_point_count = ksp.expected_points_source(duration)[0]
    self.assertGreaterEqual(expected_point_count, 0.0)

    starter_points = ksp.starter_points_source(duration)
    self.assertEqual(len(starter_points), expected_point_count)

    theta = [deltas[0]]
    for d in deltas[1:]:
      theta.append(theta[-1] + d)
    res = ksp.lnlike(theta)
    self.assertTrue(np.isfinite(res))
