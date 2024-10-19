import kagglehub

# Download latest version
path = kagglehub.dataset_download("zvr842/international-portsheds")

print("Path to dataset files:", path)

import pandas as pd
from math import radians, cos, sin, sqrt, atan2

# Function to calculate distance using Haversine formula
# Calculates the great-circle distance between two points given their latitudes and longitudes
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load port data from CSV (skip the first row if it contains headers)
df = pd.read_csv('major_ports.csv', header=None, names=['id', 'name', 'latitude', 'longitude', 'type', 'country_code', 'region'], skiprows=1)

# Convert latitude and longitude columns to float
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')  # Handle invalid strings
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop rows with invalid latitude/longitude
df = df.dropna(subset=['latitude', 'longitude'])

# Function to find the closest port
def find_closest_port(user_lat, user_lon):
    df['distance'] = df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1)
    closest_port = df.loc[df['distance'].idxmin()]
    return closest_port

# User input
user_lat = float(input("Enter your latitude: "))
user_lon = float(input("Enter your longitude: "))

# Find and display the closest port
closest_port = find_closest_port(user_lat, user_lon)
print(f"The closest port is {closest_port['name']} in {closest_port['country_code']} at a distance of {closest_port['distance']:.2f} km.")
