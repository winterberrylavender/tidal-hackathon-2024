# Ocean trash tracker
A web application that predicts the location of ocean trash based on user inputs (date, time, latitude, and longitude). It displays a map with markers for the user's chosen location and the nearest port. The application is built using Flask for the backend and Folium/Leaflet for map visualization on the frontend.
# File Descriptions
  app.py:
    A Flask application that serves the main web page and handles API requests.
    It includes two main routes:
      GET /: Serves the index.html page with an embedded map.
      POST /predict: Accepts JSON data with user-input coordinates and returns the predicted ocean trash location, estimated trash amount, and the       nearest port's coordinates.
It uses port_data.py to import port coordinates for use in the prediction response.
port_data.py:
Contains the latitude and longitude of a specific port location.
Example content:
