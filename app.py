# app.py
from flask import Flask, render_template, request, jsonify
from predictor import AgriculturalPricePredictor
from crop_predictor import CropPredictor

app = Flask(__name__)

# Instantiate both predictors
agri_predictor = AgriculturalPricePredictor()
crop_predictor = CropPredictor()

# Routes for existing agri price predictor UI
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_options')
def get_options():
    commodity = request.args.get('commodity')
    district = request.args.get('district')

    if not commodity:
        return jsonify({'commodities': agri_predictor.get_commodities()})
    elif not district:
        return jsonify({'districts': agri_predictor.get_districts(commodity)})
    else:
        return jsonify({'markets': agri_predictor.get_markets(district, commodity)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = agri_predictor.predict_prices(
        data['district'], data['commodity'], data.get('market')
    )
    return jsonify(result)

@app.route('/historical', methods=['GET'])
def historical():
    commodity = request.args.get('commodity')
    district = request.args.get('district')
    market = request.args.get('market')

    history = agri_predictor.get_historical_data(district, commodity, market)
    return jsonify(history)

# New route from FastAPI (Crop prediction)
@app.route('/api/predict_crop', methods=['POST'])
def predict_crop():
    data = request.get_json()
    try:
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pH = float(data['pH'])
        rainfall = float(data['rainfall'])

        crop = crop_predictor.predict(N, P, K, temperature, humidity, pH, rainfall)
        return jsonify({'crop': crop})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
