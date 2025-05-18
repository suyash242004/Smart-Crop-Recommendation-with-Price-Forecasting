import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class AgriculturalPricePredictor:
    def __init__(self, file_path='historical_prices.csv'):
        self.df = pd.read_csv(file_path)
        self.process_data()

    def process_data(self):
        self.df['Price Date'] = pd.to_datetime(self.df['Price Date'], format='%d-%b-%y')
        self.df = self.df.sort_values('Price Date')
        price_cols = ['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']
        for col in price_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def get_commodities(self):
        return sorted(self.df['Commodity'].unique())

    def get_districts(self, commodity):
        return sorted(self.df[self.df['Commodity'] == commodity]['District Name'].unique())

    def get_markets(self, district, commodity):
        mask = (self.df['District Name'] == district) & (self.df['Commodity'] == commodity)
        return sorted(self.df[mask]['Market Name'].unique())


    def check_stationarity(self, ts):
        return adfuller(ts.dropna())[1] <= 0.05

    def make_stationary(self, ts):
        if self.check_stationarity(ts):
            return ts, 0
        diff1 = ts.diff().dropna()
        if self.check_stationarity(diff1):
            return diff1, 1
        diff2 = diff1.diff().dropna()
        return diff2, 2

    def determine_p_q(self, series):
        return 1, 1
    

    def predict_prices(self, district_name, commodity, market_name=None, forecast_periods=1):
        mask = (self.df['District Name'] == district_name) & (self.df['Commodity'] == commodity)
        if market_name:
            mask &= (self.df['Market Name'] == market_name)

        data = self.df[mask]
        if data.empty:
            return {"error": "No data found."}

        ts_data = data.groupby('Price Date').agg({
            'Min Price (Rs./Quintal)': 'mean',
            'Max Price (Rs./Quintal)': 'mean',
            'Modal Price (Rs./Quintal)': 'mean'
        })

        results = {}
        for col in ts_data.columns:
            series = ts_data[col].fillna(method='ffill')
            
            if len(series) < 4:
                results[col.split(' ')[0]] = round(series.iloc[-1])
                continue
            
            stationary, d = self.make_stationary(series)
            p, q = self.determine_p_q(stationary)
            try:
                model = ARIMA(series, order=(p, d, q)).fit()
                forecast = model.forecast(steps=forecast_periods)
                results[col.split(' ')[0]] = round(forecast.iloc[0])
            except:
                results[col.split(' ')[0]] = round(series.rolling(window=3).mean().iloc[-1])
        return {
            'min_price': results['Min'],
            'max_price': results['Max'],
            'modal_price': results['Modal']
        }
    
    
    
    def get_historical_data(self, district, commodity, market=None):
        mask = (self.df['District Name'] == district) & (self.df['Commodity'] == commodity)
        if market:
            mask &= self.df['Market Name'] == market

        filtered = self.df[mask]

        if filtered.empty:
            return []

        data = filtered.groupby('Price Date').agg({
        'Min Price (Rs./Quintal)': 'mean',
        'Max Price (Rs./Quintal)': 'mean',
        'Modal Price (Rs./Quintal)': 'mean'
        }).reset_index()

    # Convert date to string for JSON serialization
        data['Price Date'] = data['Price Date'].dt.strftime('%Y-%m-%d')

        return data.to_dict(orient='records')

        
