import datetime
import json

import endpoints
from protorpc import messages
from google.appengine.ext import ndb

from endpoints_proto_datastore.ndb import EndpointsModel
from endpoints_proto_datastore.ndb import EndpointsAliasProperty


class TreeStatistics(EndpointsModel):
  status_app = ndb.StringProperty()
  closures_per_second = ndb.FloatProperty()
  closure_length_exponent = ndb.FloatProperty()
  generated = ndb.DateTimeProperty(auto_now_add=True)


class CQLoadSegment(EndpointsModel):
  segment = ndb.IntegerProperty()
  segment_start = ndb.StringProperty()
  request_count = ndb.IntegerProperty()
  segment_count = ndb.IntegerProperty()
  requests_per_second = ndb.FloatProperty()


class CQRequestLoad(EndpointsModel):
  project = ndb.StringProperty()
  segment_length_minutes = ndb.IntegerProperty()
  periodicity = ndb.StringProperty()
  segments = ndb.StructuredProperty(CQLoadSegment, repeated=True)
  generated = ndb.DateTimeProperty(auto_now_add=True)


class MasterList(EndpointsModel):
  masters = ndb.StringProperty(repeated=True)
  generated = ndb.DateTimeProperty(auto_now_add=True)


class BuilderBotMapping(EndpointsModel):
  builder = ndb.StringProperty()
  bots = ndb.StringProperty(repeated=True)


class MasterBotMapping(EndpointsModel):
  master = ndb.StringProperty()
  builders = ndb.LocalStructuredProperty(BuilderBotMapping, repeated=True)
  sample_time = ndb.DateTimeProperty()

  def _pre_put_hook(self):
    self.key = ndb.Key(MasterBotMapping,
        self.unique_id(self.sample_time, self.master))

  @classmethod
  def get_key_from_id(cls, sample_time, master):
    return ndb.Key(cls, cls.unique_id(sample_time, master))

  @staticmethod
  def unique_id(sample_time, master):
    return json.dumps(
        {'sample_time': sample_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
         'master': master,
        },
        sort_keys=True)


class BotConnectionStatus(EndpointsModel):
  bot = ndb.StringProperty()
  master = ndb.StringProperty()
  connected = ndb.BooleanProperty()
  running_build = ndb.BooleanProperty()
  sample_time = ndb.DateTimeProperty()

  @classmethod
  def get_key_from_id(cls, sample_time, master, bot):
    return ndb.Key(cls, cls.unique_id(sample_time, master, bot))

  @staticmethod
  def unique_id(sample_time, master, bot):
    return json.dumps(
        {'sample_time': sample_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
         'master': master,
         'bot': bot,
        },
        sort_keys=True)

  def _pre_put_hook(self):
    self.key = ndb.Key(BotConnectionStatus, self.unique_id(
      self.sample_time,
      self.master,
      self.bot))

 
class BotConnectionStatusLatest(BotConnectionStatus):
  _message_fields_schema = ('bot', 'master', 'connected', 'running_build',
                            'sample_time', 'last_connected',
                            'last_running_build',
                            'last_connected_to_last_sampled')
  last_connected = ndb.DateTimeProperty()
  last_connected_to_last_sampled = ndb.FloatProperty()
  last_running_build = ndb.BooleanProperty()

  @classmethod
  def get_key_from_id(cls, bot):
    return ndb.Key(cls, cls.unique_id(bot))

  @staticmethod
  def unique_id(bot):
    return json.dumps(
        {
         'bot': bot,
        },
        sort_keys=True)

  def _pre_put_hook(self):
    self.key = ndb.Key(BotConnectionStatusLatest, self.unique_id(
      self.bot))

  def minutes_threshold_set(self, value):
    if value < 1:
      raise endpoints.BadRequestException(
          'minutes_threshold should be at least 1')

    time = value * 60

    self._endpoints_query_info._filters.add(
        BotConnectionStatusLatest.last_connected_to_last_sampled >= time)

  @EndpointsAliasProperty(
      name='minutes_threshold',
      setter=minutes_threshold_set,
      property_type=messages.IntegerField,
      default=30)
  def minutes_threshold(self):
    raise endpoints.BadRequestException(
        'minutes_threshold value should never be accessed.')

class BotDisconnection(EndpointsModel):
  bot = ndb.StringProperty()
  master = ndb.StringProperty()
  start_time = ndb.DateTimeProperty()
  duration = ndb.FloatProperty()
  while_idle = ndb.BooleanProperty()
  generated = ndb.DateTimeProperty(auto_now_add=True)

  @classmethod
  def get_key_from_id(cls, start_time, bot):
    return ndb.Key(cls, cls.unique_id(start_time, bot))

  @staticmethod
  def unique_id(start_time, bot):
    return json.dumps(
        {'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
         'bot': bot,
        },
        sort_keys=True)

  def _pre_put_hook(self):
    self.key = ndb.Key(BotDisconnection, self.unique_id(
      self.start_time,
      self.bot))
