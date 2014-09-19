import webapp2

import views

app = webapp2.WSGIApplication([
    ('/_ah/warmup', views.StartPage),
    ('/_ah/start', views.StartPage),
    ('/admin/crawl_trees', views.CrawlTrees),
    ('/admin/crawl_cq', views.CrawlCQ),
    ('/admin/crawl_bots', views.CrawlBots),
    ('/', views.MainPage),
])
