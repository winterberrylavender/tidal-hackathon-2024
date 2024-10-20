from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

import ml_interact
# Load the model
#from tensorflow.python.keras.engine.training_v1 import Model

import port_data
import folium
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    # Here you can add your logic to process the input data
    date = request.json.get('date')
    time = request.json.get('time')
    latitude = float(request.json.get('latitude'))
    longitude = float(request.json.get('longitude'))
    close = port_data.find_closest_port(latitude, longitude)
    port_latitude=float(close['latitude'])
    port_longitude=float(close['longitude'])
    port_distance=float(close['distance'])
    trash_latitude, trash_longitude = ml_interact.interact(latitude,longitude)
    trash_distance=port_data.haversine(port_latitude,port_longitude,float(trash_latitude),float(trash_longitude))
    # Simulated result for demonstration purposes
    result = {
        'portLocation': close['name'],
        'portLatitude': port_latitude,
        'portLongitude': port_longitude,
        'portDistance' : port_distance,
        'trashLatitude': trash_latitude,
        'trashLongitude': trash_longitude,
        'trashDistance': trash_distance,
    }

    return jsonify(result)

'''@app.route('/')
def index():
    # Create a map object centered on a specific location
    map_obj = folium.Map(location=[predict().port_latitude, predict().port_longitude], zoom_start=12)

    # Add a marker to the map
    folium.Marker(
        location=[predict().port_latitude, predict().port_longitude],
        popup="This is a marker!"
    ).add_to(map_obj)

    # Save the map as an HTML file
    map_html = map_obj._repr_html_()

    # Pass the HTML representation of the map to the template
    return render_template('index.html', map_html=map_html)
'''
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)