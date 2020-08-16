import json
import requests

class APIClient:
  """Simple client for the Covid19 Tracking project."""

  CONFIG = {
    'states': {'url': '/states', 'filter': 'state'},
    'states_daily': {'url': '/states/daily', 'filter': 'state'},
    'states_info': {'url': '/states/info', 'filter': 'state'},
    'us': {'url': '/us'},
    'us_daily': {'url': '/us/daily'},
    'counties': {'url': '/counties', 'filter': 'state'}
  }

  def __init__(self):
    self._cache = {}

  def clear(self):
    """Since the data is not updated constantly, each instance will cache the data. Call this method to force it to load new data."""
    self._cache = {}

  def states(self, *states):
    """Return totals for all states. If `states` is provided, get totals for only specific states."""
    return self._get('states', states)

  def states_daily(self, *states):
    """Return daily totals for all states. If `states` is provided, get daily totals for only specific states."""
    return self._get('states_daily', states)

  def states_info(self, *states):
    """Return information for all states. If `states` is provided, get information for only specific states."""
    return self._get('states_info', states)

  def us(self):
    """Return current totals for the United States."""
    return self._get('us')

  def us_daily(self):
    """Return daily totals for the United States."""
    return self._get('us_daily')

  def counties(self, *states):
    """Return current totals for certain US counties."""
    return self._get('counties', states)

  def _get(self, key, filter_by=None):
    config = self.CONFIG[key]
    url = "https://covidtracking.com/api%s" % config['url']
    if key not in self._cache:
      self._cache[key] = requests.get(url).json()

    if filter_by:
      fKey = config['filter']
      return list(filter(lambda x: x[fKey] in filter_by, self._cache[key]))
    else:
      return self._cache[key]
