import gpxpy
import pandas as pd
import pickle

gpx_file = open("../../data/geolife/geolife_gpx/12365586.gpx", "r")

gpx = gpxpy.parse(gpx_file)

points = []

for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            points.append({
                "latitude": point.latitude,
                "longitude": point.longitude,
                "elevation": point.elevation,
                "time": point.time
            })

df = pd.DataFrame(points)

# Save as pickle
with open("../../data/geolife/geolife_gpx/12365586.pkl", "wb") as f:
    pickle.dump(df, f)

print(df.head())