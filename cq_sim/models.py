from google.appengine.ext import ndb

from endpoints_proto_datastore.ndb import EndpointsModel


class TreeStatistics(EndpointsModel):
  status_app = ndb.StringProperty()
  closures_per_second = ndb.FloatProperty()
  closure_length_exponent = ndb.FloatProperty()
  generated = ndb.DateTimeProperty(auto_now_add=True)


class CQLoadSegment(EndpointsModel):
  segment = ndb.IntegerProperty()
  request_count = ndb.IntegerProperty()
  segment_count = ndb.IntegerProperty()
  requests_per_second = ndb.FloatProperty()


class CQRequestLoad(EndpointsModel):
  project = ndb.StringProperty()
  segment_length = ndb.IntegerProperty()
  periodicity = ndb.StringProperty()
  segments = ndb.StructuredProperty(CQLoadSegment, repeated=True)
  
  generated = ndb.DateTimeProperty(auto_now_add=True)
