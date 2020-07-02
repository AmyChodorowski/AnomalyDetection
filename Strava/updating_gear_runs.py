import datetime as dt
import requests


from Strava.StravaAPI import StravaApi


def print_gear(gears):
    for gear_id in gears:
        url = '/'.join([strava.URL, 'gear', str(gear_id)])
        gear = requests.get(url, headers=strava.header).json()
        print(f"{gear['id']} clocked {gear['distance']} m")


strava = StravaApi()

# Runs to update gear
start = dt.datetime(2019, 4, 13)  # 13th April 2019
end = dt.datetime(2020, 7, 1)  # 1st July 2020
runs = strava.get_runs(start=start, end=end)

# Gear g3079004 - 'Brooks Glycerin'
# Gear g6474974 - 'ASICS GEL-EXCITE 6'
gears = ['g3079004', 'g6474974']
print_gear(gears)
gear_update = gears[1]

print()

payload = {'gear_id': gear_update}

for run in runs:
    print(f"Updating activity {run['name']} on {run['start_date']} with gear {gear_update}")

    # Update
    run_id = run['id']
    url = '/'.join([strava.URL, 'activities', str(run_id)])
    update = requests.put(url, headers=strava.header, data=payload)

print()

print_gear(gears)

print()
