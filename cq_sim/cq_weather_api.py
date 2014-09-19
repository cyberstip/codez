import datetime

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
    return query.order(-models.CQRequestLoad.generated)

  @models.MasterBotMapping.query_method(path='bot.mapping', name='bot.mapping',
      query_fields=('limit', 'pageToken', 'master',))
  def get_bot_mapping(self, query):
    """List latest crawled builder->bot mapping."""
    return query.order(-models.MasterBotMapping.sample_time)

  @models.MasterList.query_method(path='master.list', name='master.list',
      query_fields=('limit', 'pageToken',))
  def get_master_list(self, query):
    """List latest crawled masters."""
    return query.order(-models.MasterList.generated)

  @models.BotConnectionStatus.query_method(
      path='bot_connection.status', name='bot_connection.status',
      query_fields=('master', 'bot', 'limit', 'pageToken',))
  def get_bot_connection_list(self, query):
    """List bot connection statuses."""
    return query.order(-models.BotConnectionStatus.sample_time)

  @models.BotDisconnection.query_method(
      path='bot_disconnection.list', name='bot_disconnection.list',
      query_fields=('master', 'bot', 'while_idle', 'limit', 'pageToken',))
  def get_bot_disconnection_list(self, query):
    """List bot disconnection sequences."""
    return query.order(-models.BotDisconnection.start_time)


  @models.BotConnectionStatusLatest.query_method(
      path='bot_disconnection.exceed_thresh',
      name='bot_disconnection.exceed_thresh',
      query_fields=('minutes_threshold', 'master', 'limit', 'pageToken',))
  def get_bot_disconnection_exceed_thresh(self, query):
    """List bots which haven't re-connected in a specified number of minutes."""
    query = query.order(
        -models.BotConnectionStatusLatest.last_connected_to_last_sampled)
    return query.filter(models.BotConnectionStatusLatest.connected == False)


APPLICATION = endpoints.api_server([CQWeatherApi])
