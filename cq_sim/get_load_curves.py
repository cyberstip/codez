import json
import requests

data = requests.get('https://cq-weather.appspot.com/_ah/api/cqweather/v1/cq.load?limit=1&project=chromium').json()
seconds = int(data['items'][0]['segment_length_minutes']) * 60.0
seconds = 1.0
res_data = [[
  float(seg['segment']) * seconds,
  [seg['requests_per_second']]]
  for seg in data['items'][0]['segments']]

with open('chromium_load_curve.json', 'w') as f:
  json.dump([res_data], f, indent=2)
