import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

from aqi_project_utils import (
    DATASET_DIR,
    PROJECT_ROOT,
    ensure_columns,
    ensure_existing_file,
    stack_sequence_groups,
)

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

BASE_FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
OPTIONAL_FEATURE_ALIASES = {
    'temperature': ['temperature', 'Temperature', 'temp', 'Temp'],
    'humidity': ['humidity', 'Humidity'],
    'windspeed': ['windspeed', 'WindSpeed', 'wind_speed', 'Wind Speed'],
}
target = 'AQI'


def prepare_feature_columns(df):
    rename_map = {}
    selected_features = list(BASE_FEATURES)

    for canonical_name, aliases in OPTIONAL_FEATURE_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical_name
                selected_features.append(canonical_name)
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    print(f"Training with features: {selected_features}")
    return df, selected_features


def load_and_preprocess_data(file_path):
    print("Loading data...")
    file_path = ensure_existing_file(file_path, "Training dataset")
    df = pd.read_csv(file_path)
    ensure_columns(df, ["StationId", "Date", target, *BASE_FEATURES], file_path.name)
    df, features = prepare_feature_columns(df)
    
    # Sort by Date, City, StationId to maintain temporal order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['StationId', 'Date']).reset_index(drop=True)
    
    # Handle missing values (forward fill then backward fill for time series)
    df[features + [target]] = df.groupby('StationId')[features + [target]].ffill().bfill()
    # Drop any remaining NaNs 
    df = df.dropna(subset=features + [target])

    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    return df, features

def create_sequences(data, scaler_X, scaler_y, features, look_back=7):
    # Scale here per station or overall. Here we do it overall for simplicity
    scaled_features = scaler_X.transform(data[features])
    scaled_target = scaler_y.transform(data[[target]])

    X, y = [], []
    for i in range(len(scaled_features) - look_back):
        X.append(scaled_features[i:(i + look_back)])
        y.append(scaled_target[i + look_back])
        
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    print("Building LSTM model...")
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    dense_features = Dense(16, activation='relu', name='feature_dense')(x)
    outputs = Dense(1, activation='linear')(dense_features)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    file_path = DATASET_DIR / "station_day.csv"
    
    # 1. Load Data
    df, features = load_and_preprocess_data(file_path)
    
    # We will treat all stations sequentially for simplicity, but ideally you 
    # create sequences per station. Let's do a simplified approach.
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaler_X.fit(df[features])
    scaler_y.fit(df[[target]])
    
    # Save scalers
    joblib.dump(scaler_X, PROJECT_ROOT / 'scaler_X.pkl')
    joblib.dump(scaler_y, PROJECT_ROOT / 'scaler_y.pkl')
    
    look_back = 7
    # For a robust approach, we process sequences per Station
    X_all, y_all = [], []
    for station, group in df.groupby('StationId'):
        if len(group) > look_back:
            X_station, y_station = create_sequences(group, scaler_X, scaler_y, features, look_back)
            X_all.append(X_station)
            y_all.append(y_station)
            
    X, y = stack_sequence_groups(X_all, y_all, "daily station data")
    
    print(f"Total sequences created: {X.shape[0]}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 2. Train LSTM
    lstm_model = build_lstm_model((look_back, len(features)))
    
    print("Training LSTM model...")
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    lstm_model.save(PROJECT_ROOT / "lstm_base_model.h5")
    
    # 3. Extract Features from LSTM for XGBoost
    print("Extracting features using LSTM intermediate layer...")
    feature_extractor = Model(inputs=lstm_model.inputs, outputs=lstm_model.get_layer('feature_dense').output)
    
    lstm_features_train = feature_extractor.predict(X_train)
    lstm_features_test = feature_extractor.predict(X_test)
    
    # 4. Train XGBoost
    print("Training XGBoost on LSTM features...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # We use y_train directly (which is scaled). Alternatively, we could unscale before training.
    # XGBoost can learn on scaled target.
    xgb_model.fit(lstm_features_train, y_train.ravel())
    
    # Save XGBoost Model
    xgb_model.save_model(PROJECT_ROOT / "xgboost_hybrid.json")
    
    # 5. Evaluate Hybrid Model
    print("Evaluating Hybrid Model (LSTM + XGBoost)...")
    y_pred_scaled = xgb_model.predict(lstm_features_test)
    
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)
    
    print("-" * 30)
    print("Hybrid Model Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    print("Model training complete. Files saved: scaler_X.pkl, scaler_y.pkl, lstm_base_model.h5, xgboost_hybrid.json")

if __name__ == "__main__":
    main()
