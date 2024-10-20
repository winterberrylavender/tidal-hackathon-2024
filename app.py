from flask import Flask, render_template, request, jsonify
import port_data
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Here you can add your logic to process the input data
    date = request.json.get('date')
    time = request.json.get('time')
    latitude = float(request.json.get('latitude'))
    longitude = float(request.json.get('longitude'))
    close=port_data.find_closest_port(latitude,longitude)

    # Simulated result for demonstration purposes
    result = {
        'patchLocation': longitude,
        'trashAmount': latitude,
        'portLocation': close['name'],
        'closest Port': close['name'],
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
