from google.appengine.ext import ndb

from endpoints_proto_datastore.ndb import EndpointsModel


class TreeStatistics(EndpointsModel):
  status_app = ndb.StringProperty()
  closures_per_second = ndb.FloatProperty()
  closure_length_exponent = ndb.FloatProperty()
  generated = ndb.DateTimeProperty(auto_now_add=True)
