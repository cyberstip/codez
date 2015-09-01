import unittest
import numpy as np

import queue_sim
import ln_func_models


class FloatListTestCase(unittest.TestCase):
  def assertListAlmostEqual(self, a, b):
    self.assertEqual(len(a), len(b))
    for aa, bb in zip(a, b):
      self.assertAlmostEqual(aa, bb)


class UnitTestQueueSim(FloatListTestCase):
  def testNullSorted(self):
    sorty, indices = queue_sim.sort_and_get_indices([])
    self.assertEqual(sorty, [])
    self.assertEqual(indices, [])

    unsorty = queue_sim.unsort([], [])
    self.assertEqual(unsorty, [])

  def testSorted(self):
    ints = [5, 2, 6, 7, 3, 4]
    sorty, indices = queue_sim.sort_and_get_indices(ints)
    self.assertEqual(sorty, sorted(sorty))
    self.assertEqual(sorty, queue_sim.same_sort(ints, indices))
    self.assertEqual(ints, queue_sim.unsort(sorty, indices))
    self.assertEqual(indices, [1, 4, 5, 0, 2, 3])

  def testConvertASToD(self):
    a = [0.4, 0.3, 5.0, 0.2, 0.1]
    s = 1.0
    k = 3

    d = queue_sim.convert_a_s_to_d(a, s, k)
    self.assertEqual(d, [2.1, 1.3, 6.0, 1.2, 1.1])

  def testConvertAMultiSToD(self):
    a = [0.4, 0.3, 5.0, 0.2, 0.1]
    s = [1.0, 1.2, 1.2, 1.3, 1.4]
    k = 3

    d = queue_sim.convert_a_multi_s_to_d(a, s, k)
    self.assertEqual(d, [2.5, 1.5, 6.2, 1.5, 1.5])

  def testConvertADToU(self):
    a = [0.4, 0.3, 5.0, 0.2, 0.1]
    d = [2.5, 1.5, 6.2, 1.5, 1.5]
    k = 3

    u = queue_sim.convert_a_d_to_u(a, d, k)
    self.assertEqual(u, [1.5, 0.3, 5.0, 0.2, 0.1])

  def testActualsToDeltas(self):
    a = [0.5, 2.5, 5.2, 7.7]
    deltas = queue_sim.actuals_to_deltas(a)
    self.assertEqual(deltas, [0.5, 2.0, 2.7, 2.5])

  def testConvertADToDeltaS(self):
    a = [0.4, 0.3, 5.0, 0.2, 0.1]
    d = [2.5, 1.5, 6.2, 1.5, 1.5]
    k = 3
    _, deltas = queue_sim.convert_a_d_to_delta_s(a, d, k)
    self.assertEqual(deltas, [1.0, 1.2, 1.2000000000000002, 1.3, 1.4])

  def testExponentialLikelihood(self):
    # stip! test this!
    self.assertFalse(np.isfinite(queue_sim.exponential_likelihood(-1.0)))
    self.assertEqual(queue_sim.exponential_likelihood(1.0), -1.0)

  def testNormalLikelihood(self):
    # stip! test this!
    self.assertFalse(np.isfinite(queue_sim.normal_likelihood(-1.0)))

    # nerp
    # self.assertEqual(queue_sim.normal_likelihood(1.0), np.log(0.5))


class UnitTestLnFuncModels(FloatListTestCase):
  def testConstantPoissonSource(self):
    rate = 10.0
    duration = 1
    cps = ln_func_models.ConstantPoissonSource(params={'rate': rate})

    expected_point_count = cps.expected_points_source(duration)[0]
    self.assertEqual(10, expected_point_count)

    starter_points = cps.starter_points_source(duration)
    self.assertEqual(
        starter_points,
        [0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001,
          0.7000000000000001, 0.8, 0.9, 1.0])

    theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    res = cps.lnlike(theta)
    self.assertGreaterEqual(res, 0.0)

  def testPeriodicPoissonSource(self):
    rates = [1.0, 2.0, 1.0, 3.0]
    periodicity = 20
    duration = 20

    pps = ln_func_models.PeriodicPoissonSource(
        params={'rates': rates, 'periodicity': periodicity})

    expected_point_count = pps.expected_points_source(duration)[0]
    self.assertEqual(expected_point_count, 35)

    starter_points = pps.starter_points_source(duration)
    theta = [1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
        9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.0 + 1.0/3.0, 15 + 2.0/3.0,
        16.0, 16.0 + 1.0/3.0, 16.0 + 2.0/3.0, 17.0, 17 + 1.0/3.0, 17 +
        2.0/3.0, 18.0, 18.0 + 1.0/3.0, 18.0 + 2.0/3.0, 19.0, 19.0 + 1.0/3.0,
        19.0 + 2.0/3.0, 20.0]
    self.assertListAlmostEqual(starter_points, theta)
    res = pps.lnlike(theta)
    self.assertGreaterEqual(res, -12.0)

  def testKServerPipeline(self):
    rate = 1.0
    duration = 10.0
    k = 3

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
    self.assertEqual(expected_point_count, 20)

    starter_points = ksp.starter_points_source(duration)
    theta = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0]
    self.assertListAlmostEqual(starter_points, theta)

    a, d = ksp.get_a_d(theta)
    self.assertEqual(a, theta[0:len(theta)/2])
    self.assertEqual(d, theta[len(theta)/2:])

    res = ksp.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  def testGaussianDelay(self):
    rate = 1.0
    duration = 10.0
    mean = 10.0
    stddev = 10.0

    gd = ln_func_models.Pipeline([{
      'name': 'constant_poisson_source',
      'params': {'rate': rate},
    },
    {
      'name': 'gaussian_delay',
      'params': {
        'mean': mean,
        'stddev': stddev,
      }
    }],
    initial=True)

    expected_point_count = gd.expected_points_source(duration)[0]
    self.assertEqual(expected_point_count, 20)

    starter_points = gd.starter_points_source(duration)
    theta = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    self.assertListAlmostEqual(starter_points, theta)

    res = gd.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  def testConstantTreeDelay(self):
    cps_rate = 1.0
    duration = 10.0
    length = 1.0
    ctd_rate = 0.5

    ctd = ln_func_models.Pipeline([{
      'name': 'constant_poisson_source',
      'params': {'rate': cps_rate},
    },
    {
      'name': 'constant_tree_delay',
      'params': {
        'length': length,
        'rate': ctd_rate
      }
    }],
    initial=True)

    expected_point_count = ctd.expected_points_source(duration)[0]
    self.assertEqual(expected_point_count, 20)

    starter_points = ctd.starter_points_source(duration)
    theta = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    self.assertListAlmostEqual(starter_points, theta)

    a, d = ctd.get_a_d(theta)
    self.assertEqual(a, theta[0:len(theta)/2])
    self.assertEqual(d,
        [1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 11.0])

    res = ctd.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  def testAmalgamator(self):
    rate = 1.0
    duration = 10.0

    amal = ln_func_models.Pipeline([{
      'name': 'amalgamator',
      'params': {
        'source_pipelines': [[{
          'name': 'constant_poisson_source',
          'params': {'rate': rate}
          }],
          [{
            'name': 'constant_poisson_source',
            'params': {'rate': rate}
          }],
        ],
      }
    }],
    initial=True)

    expected_point_count = amal.expected_points_source(duration)[0]
    self.assertEqual(expected_point_count, 20)

    starter_points = amal.starter_points_source(duration)
    theta = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    self.assertListAlmostEqual(starter_points, theta)

    a, d = amal.get_a_d(theta)
    self.assertEqual(a, theta)
    self.assertEqual(d, theta)

    res = amal.lnlike(theta)
    self.assertTrue(np.isfinite(res))

  def testKServerAmalgamator(self):
    rate = 1.0
    duration = 10.0
    k = 3
    mean_1 = 5.0
    mean_2 = 3.0

    k_amal = ln_func_models.Pipeline([{
      'name': 'k_server_amalgamator',
      'params': {
        'servers': k,
        'source_pipelines': [[{
          'name': 'constant_poisson_source',
          'params': {'rate': rate}
          }],
          [{
            'name': 'constant_poisson_source',
            'params': {'rate': rate}
          }],
        ],
        'process_pipelines': [[{
            'name': 'gaussian_delay',
            'params': {
              'mean': mean_1,
              'stddev': 1.0,
              }
          }],
          [{
            'name': 'gaussian_delay',
            'params': {
              'mean': mean_2,
              'stddev': 1.0,
            }
          }],
        ],
      }
    }],
    initial=True)

    expected_point_count = k_amal.expected_points_source(duration)[0]

    #delete me
    starter_points = k_amal.starter_points_source(duration)
    print 'flarp', starter_points

    self.assertEqual(expected_point_count, 40)

    starter_points = k_amal.starter_points_source(duration)
    print 'yarp', starter_points
    theta = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    self.assertListAlmostEqual(starter_points, theta)

    a, d = k_amal.get_a_d(theta)
    self.assertEqual(a, theta)
    self.assertEqual(d, theta)

    res = k_amal.lnlike(theta)
    self.assertTrue(np.isfinite(res))
