# Copyright (c) 2018 Remi Salmon
# https://github.com/remisalmon/Strava-local-heatmap


import os
import time
import urllib.error
import urllib.request

import numpy as np
import matplotlib.pyplot as plt


class StravaHeatmap:

    def __init__(self, coordinates, num_activites,
                 athlete_id='Amy', location='Cambridge', activity_type='Run',
                 sigma_pixels=2,
                 use_cumululative_distribution=True,
                 reduce_points=40):

        # Globals
        self.PLT_COLORMAP = 'hot'  # matplotlib color map (see https://matplotlib.org/examples/color/colormaps_reference.html)
        self.MAX_TILE_COUNT = 100  # maximum number of OSM tiles to download
        self.OSM_TILE_SERVER = 'https://maps.wikimedia.org/osm-intl/{}/{}/{}.png'  # OSM tiles url (see https://wiki.openstreetmap.org/wiki/Tile_servers)
        self.OSM_TILE_SIZE = 256  # OSM tile size in pixels
        self.OSM_MAX_ZOOM = 19  # OSM max zoom level

        # Arguments
        self.coordinates = coordinates
        self.num_activites = num_activites
        self.athlete_id = athlete_id
        self.location = location
        self.activity_type = activity_type

        self.sigma_pixels = sigma_pixels
        self.use_cumululative_distribution = use_cumululative_distribution
        self.reduce_points = reduce_points

        self.heatmap_zoom = None
        self.tile_count = None
        self.x_tile_min = None
        self.x_tile_max = None
        self.y_tile_min = None
        self.y_tile_max = None
        self.supertile_overlay = None

    def determine_tiles(self):
        print("Determine tiles...")

        # Find min, max tile x,y coordinates
        lat_min = self.coordinates[:, 0].min()
        lat_max = self.coordinates[:, 0].max()
        lon_min = self.coordinates[:, 1].min()
        lon_max = self.coordinates[:, 1].max()

        x_tile_min = None
        x_tile_max = None
        y_tile_min = None
        y_tile_max = None

        heatmap_zoom = self.OSM_MAX_ZOOM + 1
        tile_count = self.MAX_TILE_COUNT + 1

        while tile_count > self.MAX_TILE_COUNT:

            heatmap_zoom -= 1

            if heatmap_zoom == 0:
                print('Error: The area to cover is too large')
                break

            x_tile_min, y_tile_max = deg2num(lat_min, lon_min, heatmap_zoom)
            x_tile_max, y_tile_min = deg2num(lat_max, lon_max, heatmap_zoom)

            tile_count = (x_tile_max - x_tile_min + 1) * (y_tile_max - y_tile_min + 1)

        self.heatmap_zoom = heatmap_zoom
        self.tile_count = tile_count
        self.x_tile_min = x_tile_min
        self.x_tile_max = x_tile_max
        self.y_tile_min = y_tile_min
        self.y_tile_max = y_tile_max

        print(f"Logging: {heatmap_zoom} zoom")
        print(f"Logging: {tile_count} tiles")
        print()

    def download_tiles(self):
        print("Download tiles...")

        if not os.path.exists('Tiles'):
            os.makedirs('Tiles')

        i = 0
        for x in range(self.x_tile_min, self.x_tile_max + 1):
            for y in range(self.y_tile_min, self.y_tile_max + 1):
                tile_url = self.OSM_TILE_SERVER.format(self.heatmap_zoom, x, y)

                tile_file = 'Tiles/tile_{}_{}_{}.png'.format(self.heatmap_zoom, x, y)

                if not os.path.isfile(tile_file):
                    i += 1

                    print('Logging: Downloading map tile {}/{}...'.format(i, self.tile_count))

                    if not download_tile(tile_url, tile_file):
                        tile_image = np.ones((self.OSM_TILE_SIZE, self.OSM_TILE_SIZE, 3))

                        plt.imsave(tile_file, tile_image)
        print()

    def create_heatmap(self):
        print("Creating heatmap...")

        try:
            cmap = plt.get_cmap(self.PLT_COLORMAP)
        except:
            exit('Error: Colormap {} does not exists'.format(self.PLT_COLORMAP))

        # Create supertile
        supertile_size = ((self.y_tile_max - self.y_tile_min + 1) * self.OSM_TILE_SIZE,
                          (self.x_tile_max - self.x_tile_min + 1) * self.OSM_TILE_SIZE, 3)

        supertile = np.zeros(supertile_size)

        for x in range(self.x_tile_min, self.x_tile_max + 1):
            for y in range(self.y_tile_min, self.y_tile_max + 1):
                tile_file = 'Tiles/tile_{}_{}_{}.png'.format(self.heatmap_zoom, x, y)

                tile = plt.imread(tile_file)

                i = y - self.y_tile_min
                j = x - self.x_tile_min

                supertile[i * self.OSM_TILE_SIZE:i * self.OSM_TILE_SIZE + self.OSM_TILE_SIZE,
                j * self.OSM_TILE_SIZE:j * self.OSM_TILE_SIZE + self.OSM_TILE_SIZE, :] = tile[:, :, :3]

        # Convert supertile to grayscale and invert colors
        supertile = 0.2126 * supertile[:, :, 0] + 0.7152 * supertile[:, :, 1] + 0.0722 * supertile[:, :, 2]
        supertile = 1 - supertile
        supertile = np.dstack((supertile, supertile, supertile))

        # Fill track points data
        data = np.zeros(supertile_size[:2])

        for lat, lon in self.coordinates:
            x, y = deg2xy(lat, lon, self.heatmap_zoom)

            i = int(np.round((y - self.y_tile_min) * self.OSM_TILE_SIZE))
            j = int(np.round((x - self.x_tile_min) * self.OSM_TILE_SIZE))

            # pixels are centered on the trackpoint
            data[i - self.sigma_pixels:i + self.sigma_pixels + 1,
            j - self.sigma_pixels:j + self.sigma_pixels + 1] += 1

        # Threshold track points accumulation to avoid large local maxima
        if self.use_cumululative_distribution:

            # (see https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Resolution_and_Scale)
            pixel_res = 156543.03 * np.cos(np.radians(np.mean(self.coordinates[:, 0]))) / (2 ** self.heatmap_zoom)

            # Track points max accumulation per pixel = 1/5 (track points/meters) * pixel_res (meters/pixel) per activity
            # Strava records trackpoints every 5 meters in average for cycling activities
            m = (1.0 / self.reduce_points) * pixel_res * self.num_activites

        else:
            m = 1.0

        # Threshold data to max accumulation of track points
        data[data > m] = m

        # Kernel density estimation = convolution with (almost-)Gaussian kernel
        # (see https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf)
        w_filter = int(np.sqrt(12.0 * self.sigma_pixels ** 2 + 1.0))
        data = box_filter(data, w_filter)

        # Normalize data to [0,1]
        data = (data - data.min()) / (data.max() - data.min())

        # Colorize data and remove background color
        data_color = cmap(data)
        data_color[(data_color == cmap(0)).all(2)] = [0.0, 0.0, 0.0, 1.0]
        data_color = data_color[:, :, :3]

        # Create color overlay
        supertile_overlay = np.zeros(supertile_size)
        for c in range(3):
            supertile_overlay[:, :, c] = (1.0 - data_color[:, :, c]) * supertile[:, :, c] + data_color[:, :, c]
        self.supertile_overlay = supertile_overlay
        print()

    def plot_heatmap(self):
        print("Plotting heatmap...")

        if self.supertile_overlay is None:
            raise ValueError("Error: There is not heatmap array, run .create_heatmap()")

        else:
            try:
                fig = plt.figure(figsize=(15, 15))

                ax = fig.add_subplot(111)
                ax.set_title(f"Athlete {self.athlete_id}, {self.location}, {self.activity_type}")
                plt.imshow(self.supertile_overlay)
                ax.set_aspect('equal')

                cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
                cax.get_xaxis().set_visible(False)
                cax.get_yaxis().set_visible(False)
                cax.patch.set_alpha(4)
                cax.set_frame_on(False)
                plt.show()

            except Exception as e:
                print('Error: Could not plot heatmap')
                print(e)

        print()

    def save_heatmap(self):
        print("Saving heatmap...")

        if self.supertile_overlay is None:
            raise ValueError("Error: There is not heatmap array, run .create_heatmap()")

        else:
            try:
                if not os.path.exists('Heatmaps'):
                    os.makedirs('Heatmaps')

                heatmap_file = f"Heatmaps/" \
                               f"{self.athlete_id}_{self.location}_{self.activity_type}_{self.use_cumululative_distribution}_{self.sigma_pixels}.png"
                plt.imsave(heatmap_file, self.supertile_overlay)

            except Exception as e:
                print('Error: Could not save heatmap')
                print(e)

        print()


def deg2num(lat_deg, lon_deg, zoom): # return OSM tile x,y from lat,lon in degrees (from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
  lat_rad = np.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)

  return xtile, ytile


def num2deg(xtile, ytile, zoom): # return lat,lon in degrees from OSM tile x,y (from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
  lat_deg = np.degrees(lat_rad)

  return lat_deg, lon_deg


def deg2xy(lat_deg, lon_deg, zoom): # return OSM global x,y coordinates from lat,lon in degrees
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n

    return x, y


def box_filter(image, w_box): # return image filtered with box filter
    box = np.ones((w_box, w_box))/(w_box**2)

    image_fft = np.fft.rfft2(image)
    box_fft = np.fft.rfft2(box, s = image.shape)

    image = np.fft.irfft2(image_fft*box_fft)

    return image


def download_tile(tile_url, tile_file): # download tile from url and save to file
    request = urllib.request.Request(tile_url, headers = {'User-Agent':'Mozilla/5.0'})

    try:
        response = urllib.request.urlopen(request)

    except urllib.error.URLError as e: # (see https://docs.python.org/3/howto/urllib2.html)
        print('ERROR downloading tile from OSM server failed')

        if hasattr(e, 'reason'):
            print(e.reason)

        elif hasattr(e, 'code'):
            print(e.code)

        return False

    else:
        with open(tile_file, 'wb') as file:
            file.write(response.read())

        time.sleep(0.1)

    return True
