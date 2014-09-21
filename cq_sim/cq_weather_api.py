import endpoints
from protorpc import remote

import models

# pylint: disable=R0201,C0322

package = 'CQWeather'


### Api methods.

@endpoints.api(name='cqweather', version='v1')
class CQWeatherApi(remote.Service):
  """CQ Weather API v1."""

  @models.TreeStatistics.query_method(path='tree.stats', name='tree.stats',
      query_fields=('limit', 'pageToken', 'status_app',))
  def get_trees(self, query):
    """List recorded tree statistics measured over time."""
    return query.order(-models.TreeStatistics.generated)

  @models.CQRequestLoad.query_method(path='cq.load', name='cq.load',
      query_fields=('limit', 'pageToken', 'project',))
  def get_cq_load(self, query):
    """List recorded tree statistics measured over time."""
    return query.order(-models.TreeStatistics.generated)


APPLICATION = endpoints.api_server([CQWeatherApi])
