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

logging.getLogger("tensorflow").setLevel(logging.ERROR)

DEFAULT_ADV_FEATURES = [
    'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
    'City_PM2.5', 'City_PM10', 'City_NO2', 'City_SO2', 'City_CO', 'City_O3',
]

class AdvHybridAQIPredictor:
    def __init__(self, lstm_model_path='lstm_advanced_model.h5', 
                 xgb_model_path='xgboost_advanced.json',
                 scaler_X_path='scaler_X_adv.pkl', 
                 scaler_y_path='scaler_y_adv.pkl'):
                 
        print("Loading advanced fusion models and scalers...")
        scaler_X_path = ensure_existing_file(scaler_X_path, "Advanced feature scaler")
        scaler_y_path = ensure_existing_file(scaler_y_path, "Advanced target scaler")
        lstm_model_path = ensure_existing_file(lstm_model_path, "Advanced LSTM model")
        xgb_model_path = ensure_existing_file(xgb_model_path, "Advanced XGBoost model")

        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        self.lstm_model = load_model(lstm_model_path, compile=False)
        self.feature_extractor = Model(inputs=self.lstm_model.inputs, 
                                       outputs=self.lstm_model.get_layer('feature_dense').output)
                                       
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_model_path)
        
        self.features = infer_model_features(self.scaler_X, DEFAULT_ADV_FEATURES)
        self.look_back = infer_look_back(self.lstm_model, 24)

        expected_feature_count = self.lstm_model.input_shape[-1]
        if len(self.features) != expected_feature_count:
            raise ValueError(
                f"Scaler features ({len(self.features)}) do not match model input width ({expected_feature_count})."
            )
        
    def predict_realtime(self, recent_data):
        recent_seq = prepare_recent_feature_frame(
            recent_data,
            self.features,
            self.look_back,
            "recent_data",
        )
        scaled_seq = self.scaler_X.transform(recent_seq)
        
        # (batch, time_steps, features)
        input_seq = scaled_seq.reshape(1, self.look_back, len(self.features))
        
        extracted_features = self.feature_extractor.predict(input_seq, verbose=0)
        scaled_prediction = self.xgb_model.predict(extracted_features)
        
        final_aqi = self.scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))
        return final_aqi[0][0]

if __name__ == '__main__':
    # Mock exactly what a 24-hr matrix across 12 features looks like
    mock_matrix = []
    
    # Generate 24 hours of synthetic recent records mimicking the fused dataset
    for i in range(24):
        # Station local pollutants + City macro-average pollutants
        row = [50+i, 90+i, 20+i, 10, 0.8, 25+i,  # Station feats
               55+i, 95+i, 22+i, 12, 0.9, 28+i]   # City feats
        mock_matrix.append(row)
        
    features_list = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'City_PM2.5', 'City_PM10', 'City_NO2', 'City_SO2', 'City_CO', 'City_O3']
    mock_data = pd.DataFrame(mock_matrix, columns=features_list)
    
    try:
        predictor = AdvHybridAQIPredictor()
        predicted_aqi = predictor.predict_realtime(mock_data)
        print(f"Predicted Fusion Real-Time AQI: {predicted_aqi:.2f}")
    except Exception as e:
        print(f"Model error: {e}")
        print("Ensure train_advanced_hybrid_model.py finished entirely before predicting.")
