import os
import requests

CLIENT_ID = os.environ['client_id']
CLIENT_SECRET = os.environ['client_secret']
REFRESH_TOKEN = os.environ['refresh_token']

# Refresh token

auth_url ="https://www.strava.com/oauth/token"
payload = {'client_id': CLIENT_ID,
           'client_secret': CLIENT_SECRET,
           'refresh_token': REFRESH_TOKEN,
           'grant_type': "refresh_token",
           'scope': 'read_all',
           'f': 'json'}
print("Requesting the token...\n")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']

URL = r"https://www.strava.com/api/v3"
header = {'Authorization': f'Bearer {access_token}'}

# Get activities

url = '/'.join([URL, 'athlete', 'activities'])

# Timeframe to update gear
epoch_after =  1555113600   # 13th April 2019
epoch_before = 1593561600   # 1st July 2020
param = {'before': epoch_before, 'after': epoch_after, 'per_page': 200, 'page': 1}
activities = requests.get(url, headers=header, params=param).json()

# Find runs
runs = []
for activity in activities:
    if activity['type'] == 'Run':
        runs.append(activity)

# Gear g6474974 - 'ASICS GEL-EXCITE 6'
# Gear g3079004 -  'Brooks Glycerin'

payload = {'gear_id': 'g6474974'}

for run in runs:
    print(f"Activity {run['name']} on {run['start_date']}")

    # Old - Commented out to reduce requests
    """
    gear_id = run['gear_id']
    url = '/'.join([URL, 'gear', str(gear_id)])
    gear = requests.get(url, headers=header).json()
    if gear:
        print(f"Old gear: {gear['id']} with {gear['distance']} m")
    """

    # Update
    run_id = run['id']
    url = '/'.join([URL, 'activities', str(run_id)])

    update = requests.put(url, headers=header, data=payload)

    # New - Commented out to reduce requests
    """
    gear_id = run['gear_id']
    url = '/'.join([URL, 'activities', str(run_id)])
    new = requests.get(url, headers=header).json()

    gear_id_new = new['gear_id']
    url = '/'.join([URL, 'gear', gear_id_new])
    gear_new = requests.get(url, headers=header).json()
    print(f"New gear: {gear_new['id']} with {gear_new['distance']} m")
    print()
    """
