# Copyright (c) 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import os
import urllib

import webapp2
from webapp2_extras import jinja2

import controller


class BaseHandler(webapp2.RequestHandler):
  """Provide a cached Jinja environment to each request."""

  def __init__(self, *args, **kwargs):
    webapp2.RequestHandler.__init__(self, *args, **kwargs)

  @staticmethod
  def jinja2_factory(app):
    template_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'templates'))
    config = {'template_path': template_dir}
    jinja = jinja2.Jinja2(app, config=config)
    return jinja

  @webapp2.cached_property
  def jinja2(self):
    # Returns a Jinja2 renderer cached in the app registry.
    return jinja2.get_jinja2(app=self.app, factory=BaseHandler.jinja2_factory)

  def render_response(self, _template, **context):
    # Renders a template and writes the result to the response.
    rv = self.jinja2.render_template(_template, **context)
    self.response.write(rv)


class StartPage(BaseHandler):
  def get(self):
    context = {'title': 'cq-weather' }
    self.render_response('main.html', **context)


class MainPage(BaseHandler):
  def get(self):
    context = {'title': 'cq-weather' }
    tree_urls = []
    for tree in controller.TREES:
      tree_urls.append((tree, urllib.quote(tree)))
    context['tree_urls'] = tree_urls
    context['projects'] = controller.PROJECTS

    self.render_response('main.html', **context)


class CrawlTrees(BaseHandler):
  def get(self):
    controller.write_trees()


class CrawlCQ(BaseHandler):
  def get(self):
    controller.write_cq_load()
