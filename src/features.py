import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def manhattan_distance(lat1, lon1, lat2, lon2):
    return (
        haversine(lat1, lon1, lat1, lon2) +
        haversine(lat1, lon1, lat2, lon1)
    )

def is_airport(lat, lon):
    return (
        (lat.between(40.62, 40.67) & lon.between(-73.82, -73.74)) |  # JFK
        (lat.between(40.75, 40.79) & lon.between(-73.90, -73.84))  # LGA
    )

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))

def add_features(df):
    df = df.copy()
    df.drop(columns=['id'], inplace=True, errors='ignore')

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    
    # Optional keeping dayofyear if needed later, but removing for now to reduce noise
    # df['dayofyear'] = df['pickup_datetime'].dt.dayofyear

    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = (
        df['hour'].between(7, 9) | df['hour'].between(17, 19)
    ).astype(int)

    df['pickup_airport'] = is_airport(df['pickup_latitude'], df['pickup_longitude']).astype(int)
    df['dropoff_airport'] = is_airport(df['dropoff_latitude'], df['dropoff_longitude']).astype(int)

    distance = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['distance'] = distance
    df['log_distance'] = np.log1p(distance)

    df['delta_lat'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['delta_lon'] = df['dropoff_longitude'] - df['pickup_longitude']

    b = bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['bearing_sin'] = np.sin(np.radians(b))
    df['bearing_cos'] = np.cos(np.radians(b))

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['log_distance_x_hour'] = df['log_distance'] * df['hour_sin']

    df['is_airport_trip'] = (
        (df['pickup_airport'] == 1) | (df['dropoff_airport'] == 1)
    ).astype(int)

    df['manhattan_km'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Adding these fields directly since they are low cardinal/easy categories
    if 'vendor_id' in df.columns:
        df['vendor_id'] = df['vendor_id'].astype(str)
        
    if 'passenger_count' in df.columns:
        df['passenger_count'] = df['passenger_count'].astype(str)
        
    if 'trip_duration' in df.columns:
        df['log_trip_duration'] = np.log1p(df['trip_duration'])

    df.drop(columns=['pickup_datetime'], inplace=True, errors='ignore')

    return df

def clean_train(df):
    df = df.copy()
    df = df[
        (df['pickup_latitude'].between(40.5, 41.0)) &
        (df['pickup_longitude'].between(-74.3, -73.6)) &
        (df['dropoff_latitude'].between(40.5, 41.0)) &
        (df['dropoff_longitude'].between(-74.3, -73.6)) &
        (df['trip_duration'] >= 60) &
        (df['trip_duration'] <= 14400) &
        (df['distance'] > 0.05) &
        (df['distance'] < 50)
    ].reset_index(drop=True)
    return df
