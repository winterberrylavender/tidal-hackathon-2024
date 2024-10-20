import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, MaxPooling1D, Flatten, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from geopy.distance import geodesic

# Gyre Locations
gyre_centers = {
    "North Pacific Gyre": (35.0, -140.0),  # Approx location in lat, lon
    "South Pacific Gyre": (-35.0, -120.0),
    "North Atlantic Gyre": (35.0, -40.0),
    "South Atlantic Gyre": (-35.0, -20.0),
    "Indian Ocean Gyre": (-35.0, 80.0)
}

# Load the dataset and only keep the required columns
def load_ocean_current_data(csv_file):
    print("Loading dataset...")
    df = pd.read_csv(csv_file, usecols=[
        'Data_URL', 'Start_Date_Time', 'End_Date_Time', 'Latitude_min(+deg_N)', 'Latitude_max(+deg_N)',
        'Longitude_min(+deg_E)', 'Longitude_max(+deg_E)'
    ], on_bad_lines='skip', low_memory=False)

    df['Start_Date_Time'] = pd.to_datetime(df['Start_Date_Time'], errors='coerce')
    df['End_Date_Time'] = pd.to_datetime(df['End_Date_Time'], errors='coerce')

    df.dropna(subset=['Start_Date_Time', 'End_Date_Time'], inplace=True)
    df.dropna(inplace=True)
    return df

# Preprocessing to prepare input features for CNN + LSTM model
def preprocess_data(df, num_timesteps=10):
    X_spatial = df[['Latitude_min(+deg_N)', 'Longitude_min(+deg_E)']].values
    X_temporal = []

    for i in range(len(df) - num_timesteps):
        X_temporal.append(df[['Latitude_min(+deg_N)', 'Longitude_min(+deg_E)']].iloc[i:i + num_timesteps].values)

    X_temporal = np.array(X_temporal)
    y = df['Latitude_min(+deg_N)'][num_timesteps:].values

    assert len(X_temporal) == len(y), f"X_temporal length {len(X_temporal)} and y length {len(y)} do not match."
    X_spatial = np.expand_dims(X_spatial[:len(y)], axis=1)  # Shape (batch_size, 1, 2)

    return X_spatial, X_temporal, y

# Haversine formula for distance
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).miles

# Detect the most circular area, even if none are fully circular
def detect_accumulation_region(df, start_lat, start_lon, radius=100):
    print("Detecting the area with the most current accumulation within a 100-mile radius...")

    # Filter data within a 100-mile radius
    df['distance'] = df.apply(
        lambda row: haversine(start_lat, start_lon, row['Latitude_min(+deg_N)'], row['Longitude_min(+deg_E)']), axis=1)
    nearby_points = df[df['distance'] <= radius]

    # If there are no points within the radius, return the closest point
    if len(nearby_points) == 0:
        print("No points found within the radius. Returning the closest point.")
        closest_point = df.loc[df['distance'].idxmin()]  # Get the closest point in the entire dataset
        center_lat = closest_point['Latitude_min(+deg_N)']
        center_lon = closest_point['Longitude_min(+deg_E)']
        return center_lat, center_lon

    # Measure accumulation by counting nearby points
    def count_nearby_points(lat, lon):
        return nearby_points.apply(
            lambda row: haversine(lat, lon, row['Latitude_min(+deg_N)'], row['Longitude_min(+deg_E)']) <= 10,
            axis=1).sum()

    # Apply the function to each point in the nearby region to count how many other points are within a 10-mile radius.
    nearby_points['accumulation'] = nearby_points.apply(
        lambda row: count_nearby_points(row['Latitude_min(+deg_N)'], row['Longitude_min(+deg_E)']), axis=1)

    # Find the point with the maximum accumulation (highest count of nearby points)
    max_accumulation_point = nearby_points.loc[nearby_points['accumulation'].idxmax()]

    center_lat = max_accumulation_point['Latitude_min(+deg_N)']
    center_lon = max_accumulation_point['Longitude_min(+deg_E)']

    print(f"Point with highest current accumulation: Latitude: {center_lat}, Longitude: {center_lon}")

    return center_lat, center_lon

# CNN + LSTM Model Definition using Functional API
def build_cnn_lstm_model(input_shape_temporal, input_shape_spatial):
    # Input for the temporal data (for LSTM)
    temporal_input = Input(shape=input_shape_temporal)

    # LSTM layers for temporal data
    x_temporal = LSTM(64, return_sequences=True)(temporal_input)
    x_temporal = LSTM(32)(x_temporal)

    # Input for the spatial data (for Conv1D)
    spatial_input = Input(shape=input_shape_spatial)

    # CNN layers for spatial data
    x_spatial = Conv1D(filters=32, kernel_size=1, activation='relu')(spatial_input)
    x_spatial = MaxPooling1D(pool_size=1)(x_spatial)
    x_spatial = Flatten()(x_spatial)

    # Concatenate both outputs from LSTM and CNN
    combined = concatenate([x_temporal, x_spatial])

    # Dense layers for final prediction
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)  # Output layer (e.g., future depth or current)

    # Create the model
    model = Model(inputs=[temporal_input, spatial_input], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

    return model

# Train, cross-validate, and save the model
def train_cnn_lstm_model(X_spatial, X_temporal, y, model_save_path='cnn_lstm_trash_model.h5'):
    # Ensure the folder 'models' exists in your project
    model_folder = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_folder, exist_ok=True)

    # Full path for the model
    full_model_save_path = os.path.join(model_folder, model_save_path)

    kfold = KFold(n_splits=3, shuffle=True)
    fold_no = 1
    mse_scores = []
    mae_scores = []

    for train_idx, test_idx in kfold.split(X_spatial):
        print(f"Training fold {fold_no}...")

        X_train_spatial, X_test_spatial = X_spatial[train_idx], X_spatial[test_idx]
        X_train_temporal, X_test_temporal = X_temporal[train_idx], X_temporal[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build and train the model
        model = build_cnn_lstm_model(input_shape_temporal=X_train_temporal.shape[1:],
                                     input_shape_spatial=X_train_spatial.shape[1:])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit([X_train_temporal, X_train_spatial], y_train, epochs=10, batch_size=400,
                  validation_data=([X_test_temporal, X_test_spatial], y_test), callbacks=[early_stopping])

        # Evaluate the model
        print(f"Evaluating the model on fold {fold_no}...")
        evaluation = model.evaluate([X_test_temporal, X_test_spatial], y_test)

        # Unpack the evaluation result: loss, mse, mae
        test_loss, test_mse, test_mae = evaluation[0], evaluation[1], evaluation[2]
        print(f"Fold {fold_no} - Loss: {test_loss}, MSE: {test_mse}, MAE: {test_mae}")

        mse_scores.append(test_mse)
        mae_scores.append(test_mae)
        fold_no += 1

    print(f"Average MSE: {np.mean(mse_scores)}")
    print(f"Average MAE: {np.mean(mae_scores)}")

    # Save the trained model
    print(f"Saving the trained model to {full_model_save_path}")
    model.save(full_model_save_path)  # Save the model as an .h5 file

# Predict and compare with known gyres
def predict_and_compare(model, df, start_lat, start_lon):
    # Detect the area with the highest current accumulation in a 100-mile radius
    center_lat, center_lon = detect_accumulation_region(df, start_lat, start_lon)

    # Compare with known gyre locations
    if center_lat and center_lon:
        distances_to_gyres = {gyre: haversine(center_lat, center_lon, lat, lon) for gyre, (lat, lon) in gyre_centers.items()}
        closest_gyre = min(distances_to_gyres, key=distances_to_gyres.get)
        distance_to_closest_gyre = distances_to_gyres[closest_gyre]
        print(f"Closest accumulation area is near {closest_gyre}, distance: {distance_to_closest_gyre} miles")
        return center_lat, center_lon, closest_gyre, distance_to_closest_gyre
    else:
        print("No accumulation area found within the 100-mile radius.")
        return None, None, None, None

# Load the data
csv_file = 'C:/Users/ishaa/PycharmProjects/tidal-hackathon-2024/combined_output.csv'
df = load_ocean_current_data(csv_file)

# Preprocess the data
X_spatial, X_temporal, y = preprocess_data(df)

# Train and save the model
# Train and save the model
train_cnn_lstm_model(X_spatial, X_temporal, y, model_save_path='cnn_lstm_trash_model.h5')

# Load the saved model
model = load_model('cnn_lstm_trash_model.h5')

# Predict and compare with gyres for a given future time and location
start_lat = 35.0  # Replace with actual starting latitude
start_lon = -140.0  # Replace with actual starting longitude

# Call the predict and compare function to find the closest gyre and circular region
center_lat, center_lon, closest_gyre, distance_to_closest_gyre = predict_and_compare(model, df, start_lat, start_lon)

# Output the results
if center_lat and center_lon:
    print(f"Predicted circular region is located at Latitude: {center_lat}, Longitude: {center_lon}")
    print(f"Closest gyre is the {closest_gyre}, at a distance of {distance_to_closest_gyre} miles")
else:
    print("No circular area was found within the specified radius.")

