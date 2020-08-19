import json
import requests

class APIClient:
  """
  Simple API client for the Covid19 Tracking project API (https://covidtracking.com/data/api).

  TODO: Incorporate pandas for JSON/CSV parsing so that data comes back in
  a more standard, structured, usable format without all the extra columns.
  For now, example of such code is available in notebooks/initialInvestigation

  TODO: Extend this to query for specific states.
  """

  ROUTES = {
    'states': {'url': '/states', 'filter': 'state'},
    'states_info': {'url': '/states/info', 'filter': 'state'},
    'us': {'url': '/us'},
  }

  def __init__(self):
    self._cache = {}

  def clear(self):
    """Class instances cache all data indefinitely, call this to clear"""
    self._cache = {}

  def states(self, *states):
    return self._get('states', states)

  def states_info(self, *states):
    return self._get('states_info', states)

  def us(self):
    return self._get('us')

  def _get(self, key, filter_by=None):
    route = self.ROUTES[key]
    url = "https://covidtracking.com/api%s" % rout['url']
    if key not in self._cache:
      self._cache[key] = requests.get(url).json()

    if filter_by:
      fKey = route['filter']
      return list(filter(lambda x: x[fKey] in filter_by, self._cache[key]))
    else:
      return self._cache[key]
