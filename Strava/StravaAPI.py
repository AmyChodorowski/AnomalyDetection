import os
import requests


class StravaApi:

    def __init__(self):
        self.CLIENT_ID = os.environ['client_id']
        self.CLIENT_SECRET = os.environ['client_secret']
        self.REFRESH_TOKEN = os.environ['refresh_token']
        self.URL = r"https://www.strava.com/api/v3"

        self.header = self.refresh_token()

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

    @staticmethod
    def convert_runs(runs_lt):
        """
        Convert the runs from list() to DataFrame
        :param runs_lt: Runs in a list of DetailedActivity
        :return: pandas.DataFrame
        """
        runs_df = pd.DataFrame()
        return runs_df

    def save_runs(self):
        pass