"""
1) Bulk download Strava data (https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#Bulk)
2) Take a note of the [ATHLETE_ID]: "export_[ATHLETE_ID]"
3) Rename "activities" folder as "[ATHLETE_ID]_activities"
4) Copy "activities.csv" into "[ATHLETE_ID]_activities" folder
5) Copy "[ATHLETE_ID]_activities" folder into "Data" folder
"""

import os
import gzip
import re

import gpxpy
import xml.etree.ElementTree as etree
import fitdecode

import pandas as pd
import datetime as dt


class StravaData:

    def __init__(self, athlete_id):
        self.athlete_id = athlete_id
        self.activity_folder, self.activities = self.get_activities()

        self.data_headers = ['timestamp',            # datetime.datetime, 2017-07-29 11:19:22+00:00
                             'position_long',        # float, degs
                             'position_lat',         # float, degs
                             'distance',             # float, m
                             'speed',                # float, m/s
                             'altitude',             # float, m
                             'heart_rate',           # int,   bpm
                             'enhanced_altitude',
                             'enhanced_speed',
                             'cadence',
                             'fractional_cadence']

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

    def get_activity_data(self, id):
        """
        Get the raw data of an activity and place into a DataFrame, using self.data_headers
        Also save under [ATHLETE_ID]_activities/[ACTIVITY_ID].csv
        :param id: ACTIVITY_ID
        :return: pandas.DataFrame
        """

        # Files types:
        #   .gpx (GPs eXchange format) - XML file, common GPS data schema
        #   .tcx (Training Center Xml) - XML file, common fitness data schema
        #   .fit (Flexible and Interoperable data Transfer) - Binary file, Garmin's verison of .tcx

        try:
            file = self.activities.loc[id, 'Filename']
            file = file.replace(r"activities/", '')
            file = file.replace(r".gz", '')

        except Exception as e:
            print(f"No data for activity {id}")
            print(e)
            print()
            return

        print(f"Getting data for activity {id}: {file}...")

        df = pd.DataFrame(columns=self.data_headers)

        try:
            path = os.path.join(self.activity_folder, file)
            if file.endswith('.gpx'):
                df = self.parse_gpx_file(df, path)
            elif file.endswith('.tcx'):
                df = self.parse_tcx_file(df, path)
            elif file.endswith('.fit'):
                df = self.parse_fit_file(df, path)

        except Exception as e:
            print(f"Unsuccessful for activity {id}: {file}")
            print(e)
            print()
            return

        # Save the file
        path = os.path.join(self.activity_folder, f"{id}.csv")
        df.to_csv(path)
        print(f"Saved under {id}.csv")
        print()
        return df

    def unzip_gz_files(self):
        for file in os.listdir(self.activity_folder):
            if file.endswith(".gz"):
                file_z = file
                file_uz = file.replace('.gz', '')
                path_z = os.path.join(self.activity_folder, file_z)
                path_uz = os.path.join(self.activity_folder, file_uz)
                print(f"Uncompress {file_z} to {file_uz}")

                with gzip.open(path_z, 'rb') as f_in:
                    file_content = f_in.read()
                    file_content = file_content.strip()
                    with open(path_uz, 'wb') as f_out:
                        f_out.write(file_content)

                # Remove gaps at the start

    @staticmethod
    def get_data_folder():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path.replace('Strava', 'Data')
        return dir_path

    @staticmethod
    def parse_gpx_file(df, path):
        # https://ocefpaf.github.io/python4oceanographers/blog/2014/08/18/gpx/
        index = len(df.index)
        gpx = gpxpy.parse(open(path))

        for track in gpx.tracks:
            for segment in track.segments:
                for point_idx, point in enumerate(segment.points):
                    df.loc[index, 'timestamp'] = point.time
                    df.loc[index, 'position_lat'] = point.latitude
                    df.loc[index, 'position_long'] = point.longitude
                    df.loc[index, 'speed'] = segment.get_speed(point_idx)
                    df.loc[index, 'altitude'] = point.elevation

                    # Update index
                    index += 1

        return df

    @staticmethod
    def parse_tcx_file(df, path):
        index = len(df.index)

        with open(path) as xml_file:
            xml_str = xml_file.read()
            xml_str = re.sub(' xmlns="[^"]+"', '', xml_str, count=1)
            tcx = etree.fromstring(xml_str)
            activities = tcx.findall('.//Activity')
            for activity in activities:
                tracking_points = activity.findall('.//Trackpoint')
                for point in list(tracking_points):
                    ts = dt.datetime.strptime(str(point.find('Time').text), '%Y-%m-%dT%H:%M:%SZ')
                    df.loc[index, 'timestamp'] = ts
                    df.loc[index, 'position_lat'] = point.find('Position').find('LatitudeDegrees').text
                    df.loc[index, 'position_long'] = point.find('Position').find('LongitudeDegrees').text
                    df.loc[index, 'distance'] = point.find('DistanceMeters').text
                    df.loc[index, 'altitude'] = point.find('AltitudeMeters').text
                    if point.find('HeartRateBpm') is not None:
                        df.loc[index, 'heart_rate'] = point.find('HeartRateBpm').find('Value').text
                    if point.find('Extensions').find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX') is not None:
                        df.loc[index, 'speed'] = point.find('Extensions').find('{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX').find('Speed').text

                    # Update index
                    index += 1

        return df

    @staticmethod
    def parse_fit_file(df, path):
        index = len(df.index)
        SEMI_2_DEGS = 180.0 / 2 ** 31

        # https://maxcandocia.com/article/2017/Sep/22/converting-garmin-fit-to-csv/
        required_fields = ['timestamp', 'position_lat', 'position_long', 'altitude']

        with fitdecode.reader.FitReader(path) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    if hasattr(frame, 'fields'):
                        fields = [f.name for f in frame.fields]
                        if all(f in fields for f in required_fields):
                            for field in frame.fields:
                                if field.name.startswith('position_'):
                                    df.loc[index, field.name] = field.value * SEMI_2_DEGS
                                else:
                                    df.loc[index, field.name] = field.value

                            # Update index
                            index += 1
            pass

        return df



