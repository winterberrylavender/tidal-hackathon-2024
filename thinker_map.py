# dont need this file
import tkinter as tk
from PIL import Image, ImageTk

# Example Map Boundaries (Replace with your map's actual latitude/longitude)
map_bounds = {
    'top_left': {'lat': 49.384358, 'lon': -124.848974},  # Top-left corner (lat, lon)
    'bottom_right': {'lat': 24.396308, 'lon': -66.93457}  # Bottom-right corner (lat, lon)
}


def calculate_lat_lon(click_x, click_y, img_width, img_height):
    """
    Calculate latitude and longitude based on click position and map dimensions.
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
    Handle click event and display the latitude and longitude.
    """
    click_x, click_y = event.x, event.y  # Get click coordinates on the image

    # Get image dimensions
    img_width = canvas.winfo_width()
    img_height = canvas.winfo_height()

    # Calculate latitude and longitude
    lat, lon = calculate_lat_lon(click_x, click_y, img_width, img_height)

    # Display the calculated latitude and longitude
    label.config(text=f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")


# Initialize Tkinter window
root = tk.Tk()
root.title("Map Click to Lat/Lon")

# Load the map image
map_image = Image.open("World_Map.jpg")  # Replace with your image file
map_photo = ImageTk.PhotoImage(map_image)

# Create a canvas to display the map
canvas = tk.Canvas(root, width=map_photo.width(), height=map_photo.height())
canvas.pack()

# Display the image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=map_photo)

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_click)

# Label to display latitude and longitude
label = tk.Label(root, text="Click on the map to get latitude and longitude")
label.pack()

# Run the Tkinter event loop
root.mainloop()
