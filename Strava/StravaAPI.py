"""
1) Create Strava API account (https://developers.strava.com/docs/getting-started/#account)
2) Take a note of the [CLIENT_ID] and [CLIENT_SECRECT]
3) Paste your [CLIENT_ID] into the URL: http://www.strava.com/oauth/authorize?client_id=[CLIENT_ID]&response_type=code&redirect_uri=http://localhostscope=activity:read_all,activity:write
4) Place the URL into browser, click "Authorize"
5) Within the redirected URL, copy the [CODE]
6) Place [CLIENT_ID], [CLIENT_SECRECT] and [CODE] into the command: curl -X POST https://www.strava.com/oauth/token -F client_id=[CLIENT_ID]	-F client_secret=[CLIENT_SECRET] -F code=[CODE]	-F grant_type=authorization_code
7) Run the command and copy the [REFRESH_TOKEN]
8) Place the [CLIENT_ID], [CLIENT_SECRECT] and [REFRESH_TOKEN] as environment variables
"""

import os
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class StravaApi:

    def __init__(self):
        self.CLIENT_ID = os.environ['CLIENT_ID']
        self.CLIENT_SECRET = os.environ['CLIENT_SECRET']
        self.REFRESH_TOKEN = os.environ['REFRESH_TOKEN']
        self.URL = r"https://www.strava.com/api/v3"

        self.header = self.refresh_token()
        self.althete_id = self.get_althete()

    def refresh_token(self):
        """
        Using the refresh the access token and place it into the header format
        :return: {'Authorization': f'Bearer {access_token}'}
        """

        auth_url = r"https://www.strava.com/oauth/token"
        payload = {'client_id': self.CLIENT_ID,
                   'client_secret': self.CLIENT_SECRET,
                   'refresh_token': self.REFRESH_TOKEN,
                   'grant_type': "refresh_token",
                   'f': 'json'}

        print("Requesting the token...\n")
        r = requests.post(auth_url, data=payload, verify=False)
        access_token = r.json()['access_token']
        header = {'Authorization': f'Bearer {access_token}'}

        return header

    def get_althete(self):
        url = '/'.join([self.URL, 'athlete'])
        athlete = requests.get(url, headers=self.header).json()
        return athlete['id']

    def get_runs(self, start, end):
        """
        Get the runs between start and end
        :param start: datetime
        :param end: datetime
        :return: list(DetailedActivity - https://developers.strava.com/docs/reference/#api-models-DetailedActivity)
        """

        # Convert datetime into epoch
        epoch_after = start.timestamp()
        epoch_before = end.timestamp()

        # Get all activities
        url = '/'.join([self.URL, 'athlete', 'activities'])
        param = {'before': epoch_before, 'after': epoch_after, 'per_page': 200, 'page': 1}
        activities = requests.get(url, headers=self.header, params=param).json()

        # Find runs
        runs = []
        for activity in activities:
            if activity['type'] == 'Run':
                runs.append(activity)

        return runs

    def convert_runs(self, runs_lt):
        """
        Convert the runs from list() to DataFrame
        :param runs_lt: Runs in a list of DetailedActivity
        :return runs_df: pandas.DataFrame
        """
        runs_df = pd.DataFrame()

        for run in runs_lt:

            run = self.flatten_run(run)
            runs_df = runs_df.append(run, ignore_index=True)

        return runs_df

    def save_runs(self, runs_df):
        """
        Save the runs to file
        :param runs_dt: pandas.DataFrame of the runs
        """
        # First and last run
        runs_df.sort_values('start_date', inplace=True)
        first, last = runs_df.start_date.min(), runs_df.start_date.max()
        first = first.replace('T', '-')
        last = last.replace('T', '-')
        first = first.replace(':', '-')
        last = last.replace(':', '-')
        first = first.replace('Z', '')
        last = last.replace('Z', '')

        # Make id str
        runs_df['id'] = runs_df['id'].apply(lambda x: int(x))
        runs_df['id'] = runs_df['id'].astype(str)

        # Make athlete id str
        runs_df['athlete_id'] = runs_df['athlete_id'].apply(lambda x: int(x))
        runs_df['athlete_id'] = runs_df['athlete_id'].astype(str)

        # Filepath
        dir_path = self.get_data_folder()
        path = os.path.join(dir_path, f"{self.althete_id}_{first}_{last}_runs.csv")

        runs_df.to_csv(path, sep=',', header=True)

    @staticmethod
    def flatten_run(run):
        """
        Make the DetailedActivity model a 1D dict
        :param run: DetailedActivity - https://developers.strava.com/docs/reference/#api-models-DetailedActivity
        :return: dict()
        """
        # Athlete
        run['athlete_id'] = run['athlete']['id']
        run.pop('athlete')

        # Start [lat, long] coordinates
        run['start_lat'] = run['start_latlng'][0]
        run['start_lng'] = run['start_latlng'][1]
        run.pop('start_latlng')

        # End [lat, long] coordinates
        run['end_lat'] = run['end_latlng'][0]
        run['end_lng'] = run['end_latlng'][1]
        run.pop('end_latlng')

        # Map
        run['map_id'] = run['map']['id']
        run['map_polyline'] = run['map']['summary_polyline']
        run.pop('map')

        return run

    @staticmethod
    def get_data_folder():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path.replace('Strava', 'Data')
        return dir_path