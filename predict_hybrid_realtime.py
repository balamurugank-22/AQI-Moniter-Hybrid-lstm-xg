import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging

import pandas as pd
import xgboost as xgb
import joblib
from keras.models import Model, load_model

from aqi_project_utils import (
    ensure_existing_file,
    infer_look_back,
    infer_model_features,
    prepare_recent_feature_frame,
)

DEFAULT_FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class RealTimeHybridAQIPredictor:
    def __init__(self, lstm_model_path='lstm_base_model.h5', 
                 xgb_model_path='xgboost_hybrid.json',
                 scaler_X_path='scaler_X.pkl', 
                 scaler_y_path='scaler_y.pkl'):
                 
        print("Loading models and scalers...")
        scaler_X_path = ensure_existing_file(scaler_X_path, "Feature scaler")
        scaler_y_path = ensure_existing_file(scaler_y_path, "Target scaler")
        lstm_model_path = ensure_existing_file(lstm_model_path, "LSTM model")
        xgb_model_path = ensure_existing_file(xgb_model_path, "XGBoost model")

        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        self.lstm_model = load_model(lstm_model_path, compile=False)
        # Recreate Feature Extractor
        self.feature_extractor = Model(inputs=self.lstm_model.inputs, 
                                       outputs=self.lstm_model.get_layer('feature_dense').output)
                                       
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_model_path)
        
        self.features = infer_model_features(self.scaler_X, DEFAULT_FEATURES)
        self.look_back = infer_look_back(self.lstm_model, 7)

        expected_feature_count = self.lstm_model.input_shape[-1]
        if len(self.features) != expected_feature_count:
            raise ValueError(
                f"Scaler features ({len(self.features)}) do not match model input width ({expected_feature_count})."
            )
        
    def predict_realtime(self, recent_data):
        """
        recent_data: A DataFrame of the most recent 'look_back' (e.g. 7) time steps/days
        Must contain columns matching self.features
        """
        recent_seq = prepare_recent_feature_frame(
            recent_data,
            self.features,
            self.look_back,
            "recent_data",
        )
        scaled_seq = self.scaler_X.transform(recent_seq)
        
        # Reshape for LSTM: (batch_size, time_steps, features)
        input_seq = scaled_seq.reshape(1, self.look_back, len(self.features))
        
        # 1. Extract Temporal Features
        extracted_features = self.feature_extractor.predict(input_seq, verbose=0)
        
        # 2. Predict with XGBoost
        scaled_prediction = self.xgb_model.predict(extracted_features)
        
        # 3. Inverse transform target
        final_aqi = self.scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))
        return final_aqi[0][0]

if __name__ == '__main__':
    # --- Example Real-Time Testing Usage ---
    
    # In real-time integration, you would pull the last 7 time records for a given station
    # Example mock data exactly like station_day.csv feature scope
    mock_data = pd.DataFrame({
        'PM2.5': [59, 29, 51, 53, 50, 48, 55],
        'PM10': [110, 77, 88, 98, 90, 85, 100],
        'NO2': [29, 38, 39, 15, 20, 22, 25],
        'SO2': [7, 5, 8, 19, 15, 12, 10],
        'CO': [0.66, 1.49, 0.59, 1.06, 0.8, 0.7, 0.9],
        'O3': [21, 32, 32, 27, 25, 28, 30],
        'temperature': [28, 29, 30, 31, 30, 29, 28],
        'humidity': [60, 58, 55, 52, 54, 57, 59],
        'windspeed': [12, 10, 11, 9, 8, 10, 11],
    })
    
    try:
        predictor = RealTimeHybridAQIPredictor()
        predicted_aqi = predictor.predict_realtime(mock_data)
        print(f"Predicted Real-Time AQI: {predicted_aqi:.2f}")
    except Exception as e:
        print(f"Model error: {e}")
        print("Run the training script (train_hybrid_model.py) entirely first to generate necessary artifacts.")
