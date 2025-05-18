# crop_predictor.py
import pickle
import numpy as np

class CropPredictor:
    def __init__(self, model_path='model.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, N, P, K, temperature, humidity, pH, rainfall):
        input_array = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        prediction = self.model.predict(input_array)
        return prediction[0]
