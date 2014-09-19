import contextlib
import datetime
import logging
import json

from google.appengine.api import urlfetch
from google.appengine.ext import deferred

import models


get_masters_url = 'https://chrome-build-extract.appspot.com/get_masters?json=1'
get_specific_master_template = ('https://chrome-build-extract.appspot.com/'
                                'get_master/%s')

def get_json(url, deadline=60):
  logging.info('crawling %s...' % url)
  result = urlfetch.fetch(url, deadline=deadline)
  if result.status_code != 200:
    raise ValueError('for url %s status was %d' % (
      url, result.status_code))
  return json.loads(result.content)


def sample_uptime_from_masters():
  masters = get_json(get_masters_url)['masters']
  #masters = masters[0:1]
  #masters = ['tryserver.chromium.linux']
  for master in masters:
    deferred.defer(sample_bot_uptime, master, _target='crawler')
  models.MasterList(masters=masters).put()


def sample_bot_uptime(master):
  url = get_specific_master_template % master
  master_json = get_json(url)

  master_timestamp = datetime.datetime.strptime(
      master_json['created'],
      '%Y-%m-%dT%H:%M:%S.%f',
  )

  mapping_record_key = models.MasterBotMapping.get_key_from_id(
      master_timestamp, master)
  if not mapping_record_key.get():
    builders = []
    for b, b_data in master_json['builders'].iteritems():
      builders.append(
          models.BuilderBotMapping(
            builder=b,
            bots=b_data['slaves'],
          )
      )
    models.MasterBotMapping(
        master=master,
        builders=builders,
        sample_time=master_timestamp,
    ).put()

  for bot in master_json['slaves']:
    bot_record_key = models.BotConnectionStatus.get_key_from_id(
        master_timestamp, master, bot)
    bot_record_obj = bot_record_key.get()
    if not bot_record_obj:
      connected = master_json['slaves'][bot]['connected']
      running_build = bool(master_json['slaves'][bot]['runningBuilds'])
      models.BotConnectionStatus(
          bot=bot,
          master=master,
          connected=connected,
          running_build=running_build,
          sample_time=master_timestamp).put()
      latest_key = models.BotConnectionStatusLatest.get_key_from_id(bot)
      latest_obj = latest_key.get()
      last_connected = None
      last_running_build = None
      if latest_obj:
        last_connected = latest_obj.last_connected
        last_running_build = latest_obj.last_running_build
        if last_connected and connected and not latest_obj.connected:
          if latest_obj.last_running_build is not None:
            models.BotDisconnection(
                bot=bot,
                master=master,
                start_time=last_connected,
                duration=(master_timestamp - last_connected).total_seconds(),
                while_idle=not latest_obj.last_running_build).put()

      last_connected = master_timestamp if connected else last_connected
      last_running_build = running_build if connected else last_running_build
      last_connected_to_last_sampled = None
      if last_connected:
        last_connected_to_last_sampled = (
            master_timestamp - last_connected).total_seconds()
      models.BotConnectionStatusLatest(
        bot=bot,
        master=master,
        connected=connected,
        running_build=last_running_build,
        sample_time=master_timestamp,
        last_connected=last_connected,
        last_connected_to_last_sampled=last_connected_to_last_sampled,
        last_running_build=last_running_build).put()


def get_masters():
  return models.MasterList.query().order(
      -models.MasterList.generated).fetch(1).masters


def get_all_bots_for_master(master):
  return models.MasterBotMapping.query(
      models.MasterBotMapping.master == master).order(
          -models.MasterBotMapping.generated).fetch(1).builders
