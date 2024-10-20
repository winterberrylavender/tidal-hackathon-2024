from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import math
# Load the model
#from tensorflow.python.keras.engine.training_v1 import Model

import port_data
#import Ml model
import folium
def interact(start_lat, start_lon):
    model = tf.keras.models.load_model('cnn_lstm_trash_model.h5')
    # Example of user input for latitude, longitude, and the temporal sequence
    #start_lat = float(input("Enter starting latitude: "))
    #start_lon = float(input("Enter starting longitude: "))

    # Assume you need to preprocess this input in the same way as during training
    # Example: Prepare spatial input (reshaped into the format expected by the model)
    X_spatial = np.array([[start_lat, start_lon]])
    X_spatial = np.expand_dims(X_spatial, axis=1)  # Reshape to match model's input

    # Example temporal input: you may need to create a sequence based on historical data
    # For demonstration, let's assume this is a sequence of past lat/lon
    X_temporal = np.array([[  # This should have 10 timesteps
        [start_lat - 0.1, start_lon + 0.1],  # Past coordinates (just for example)
        [start_lat - 0.08, start_lon + 0.08],
        [start_lat - 0.06, start_lon + 0.06],
        [start_lat - 0.05, start_lon + 0.05],
        [start_lat - 0.04, start_lon + 0.04],
        [start_lat - 0.03, start_lon + 0.03],
        [start_lat - 0.02, start_lon + 0.02],
        [start_lat - 0.01, start_lon + 0.01],
        [start_lat, start_lon],  # Current coordinates
        [start_lat + 0.01, start_lon - 0.01]  # Future or next coordinate (just for example)
    ]])

    # Ensure temporal input is reshaped appropriately for the LSTM layer
    X_temporal = np.array(X_temporal)  # This needs to match how you trained the model

    # Predict using the loaded model
    prediction = model.predict([X_temporal, X_spatial])

    # Accessing the predicted latitude and longitude from the prediction
    predicted_latitude = prediction[0][0]  # First value: latitude
    predicted_longitude = (start_lon - (0.7(start_lat-predicted_latitude))+1)  # Second value: longitude

    # Convert them to strings
    trash_latitude_str = str(predicted_latitude)
    trash_longitude_str = str(predicted_longitude)
    return trash_latitude_str, trash_longitude_str
