import os

from Strava.StravaData import StravaData

# Audrey - 18301580
# Ethan - 22985222
# Amy - 23312763

athletes = [23312763, 18301580, 22985222]

for athlete in athletes:

    strava = StravaData(athlete_id=athlete)
    print("###################################################")
    print(f"Athlete {athlete}")
    print()

    strava.unzip_gz_files()

    # 1122298786 = 1122298786.gpx
    # strava.get_activity_data(1122298786)

    # 1496404022 = 1611726874.tcx
    # strava.get_activity_data(1496404022)

    # 3673509552 = 3922240387.fit
    # strava.get_activity_data(3673509552)

    """
    for id in strava.activities.index:
        strava.get_activity_data(id)
    """

    for id in strava.activities.index:
        file = os.path.join(strava.activity_folder, f"{id}.csv")
        if not os.path.exists(file):
            strava.get_activity_data(id)

    print()
    print()




