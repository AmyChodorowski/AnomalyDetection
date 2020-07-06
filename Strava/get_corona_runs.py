import datetime as dt

from Strava.StravaAPI import StravaApi

strava = StravaApi()

# Get runs
start = dt.datetime(2020, 3, 20)  # 20th March 2020
end = dt.datetime(2020, 7, 1)     # 1st July 2020
runs = strava.get_runs(start=start, end=end)
runs = strava.convert_runs(runs)
strava.save_runs(runs)