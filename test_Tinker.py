#dont need this file
import tkinter as tk
from PIL import Image, ImageTk
import os
print(os.getcwd())
import virtualenv

# Define the map boundaries (in terms of latitude and longitude)
# Example values, modify based on your map image
map_bounds = {
    'top_left': {'lat': 90.0, 'lon': -180.0},  # Top-left corner (North-West, lat 90, lon -180)
    'bottom_right': {'lat': -90.0, 'lon': 180.0}  # Bottom-right corner (South-East, lat -90, lon 180)
}


def calculate_lat_lon(click_x, click_y, img_width, img_height):
    """
    Convert the clicked pixel position to latitude and longitude based on map dimensions and boundaries.
    """
    # Calculate latitude
    lat = map_bounds['top_left']['lat'] - (click_y / img_height) * (
                map_bounds['top_left']['lat'] - map_bounds['bottom_right']['lat'])

    # Calculate longitude
    lon = map_bounds['top_left']['lon'] + (click_x / img_width) * (
                map_bounds['bottom_right']['lon'] - map_bounds['top_left']['lon'])

    return lat, lon


def on_click(event):
    """
    Handle the click event on the map image and display the corresponding latitude and longitude.
    """
    click_x, click_y = event.x, event.y  # Get the coordinates of the click on the image

    # Get the image dimensions
    img_width = canvas.winfo_width()
    img_height = canvas.winfo_height()

    # Calculate latitude and longitude
    lat, lon = calculate_lat_lon(click_x, click_y, img_width, img_height)

    # Update the label with the latitude and longitude
    label.config(text=f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")


# Initialize the Tkinter window
root = tk.Tk()
root.title("World Map Latitude/Longitude Finder")

# Load the world map image using Pillow
map_image = Image.open("World_Map.jpg")  # Replace with the path to your map image
map_photo = ImageTk.PhotoImage(map_image)

# Create a Canvas widget to display the map image
canvas = tk.Canvas(root, width=map_photo.width(), height=map_photo.height())
canvas.pack()

# Display the image on the Canvas
canvas.create_image(0, 0, anchor=tk.NW, image=map_photo)

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_click)

# Create a Label widget to show the latitude and longitude
label = tk.Label(root, text="Click on the map to get latitude and longitude")
label.pack()

# Run the Tkinter event loop
root.mainloop()
