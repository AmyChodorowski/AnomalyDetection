"""
1) Bulk download Strava data (https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#Bulk)
2) Take a note of the [ATHLETE_ID]: "export_[ATHLETE_ID]"
3) Rename "activities" folder as "[ATHLETE_ID]_activities"
4) Copy "activities.csv" into "[ATHLETE_ID]_activities" folder
5) Copy "[ATHLETE_ID]_activities" folder into "Data" folder
"""

import os
import gzip
import shutil

import gpxpy
import fitdecode

import pandas as pd


class StravaData:

    def __init__(self, athlete_id):
        self.athlete_id = athlete_id
        self.activity_folder, self.activities = self.get_activities()

        self.data_headers = ['timestamp',      # int,   seconds
                             'position_lat',   # float, degs
                             'position_long',  # float, degs
                             'speed'           # Float, m/s
                             'elevation']      # Float, m

    def get_activities(self):
        """
        Get the list of activities and place into a DataFrame
        Also get the locations of the activities
        :return: [str, pandas.DataFrame]
        """
        dir_path = self.get_data_folder()
        folder = os.path.join(dir_path, f"{self.athlete_id}_activities")
        path = os.path.join(folder, "activities.csv")
        activities_df = pd.read_csv(path, header=0)
        activities_df = activities_df.set_index('Activity ID')
        return folder, activities_df

    # TODO
    def get_activity_data(self, id):
        """
        Get the raw data of an activity and place into a DataFrame, using self.data_headers
        Also save under [ATHLETE_ID]_df/[ACTIVITY_ID].csv
        :param id: ACTIVITY_ID
        :return: pandas.DataFrame
        """

        print(f"Getting data for activity {id}...")
        df = pd.DataFrame(columns=self.data_headers)

        # Files types:
        #   .gpx (GPs eXchange format) - XML file, common GPS data schema
        #   .tcx (Training Center Xml) - XML file, common fitness data schema
        #   .fit (Flexible and Interoperable data Transfer) - Binary file, Garmin's verison of .tcx

        file = self.activities.loc[id, 'Filename']
        file = file.replace(r"activities/", '')
        file = file.replace(r".gz", '')

        path = os.path.join(self.activity_folder, file)
        if file.endswith('.gpx'):
            df = self.parse_gpx_file(df, path)
        elif file.endswith('.tcx'):
            df = self.parse_tcx_file(df, path)
        elif file.endswith('.fit'):
            df = self.parse_fit_file(df, path)

        # Save the file
        df.to_csv()

        return df

    def unzip_gz_files(self):
        for file in os.listdir(self.activity_folder):
            if file.endswith(".gz"):
                print(f"Uncompress {file} to {file.replace('.gz', '')}")
                path = os.path.join(self.activity_folder, file)
                with gzip.open(path, 'rb') as f_in:
                    with open(path.replace('.gz', ''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def get_data_folder():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path.replace('Strava', 'Data')
        return dir_path

    # TODO
    @staticmethod
    def parse_gpx_file(df, path):
        gpx = gpxpy.parse(open(path))
        for track in gpx.tracks:
            for segment in track.segments:
                for point_idx, point in enumerate(segment.points):
                    

        print("{} track(s)".format(len(gpx.tracks)))
        track = gpx.tracks[0]

        print("{} segment(s)".format(len(track.segments)))
        segment = track.segments[0]

        print("{} point(s)".format(len(segment.points)))
        return df

    # TODO
    @staticmethod
    def parse_tcx_file(df, path):
        return df

    # TODO
    @staticmethod
    def parse_fit_file(df, path):
        allowed_fields = ['timestamp', 'position_lat', 'position_long', 'distance',
                          'enhanced_altitude', 'altitude', 'enhanced_speed',
                          'speed', 'heart_rate', 'cadence', 'fractional_cadence']
        required_fields = ['timestamp', 'position_lat', 'position_long', 'altitude']
        with fitdecode.reader.FitReader(path) as fit:
            for frame in fit:
                # The yielded frame object is of one of the following types:
                # * fitdecode.FitHeader
                # * fitdecode.FitDefinitionMessage
                # * fitdecode.FitDataMessage
                # * fitdecode.FitCRC

                if isinstance(frame, fitdecode.FitDataMessage):
                    # Here, frame is a FitDataMessage object.
                    # A FitDataMessage object contains decoded values that
                    # are directly usable in your script logic.
                    f = []
                    for field in frame.fields:
                        f.append(str(field.name))
                        pass
                    print(frame.name, f)
        print('Finished')

        return df



