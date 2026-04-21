
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

# Define columns
station_feats = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
city_feats = ['City_PM2.5', 'City_PM10', 'City_NO2', 'City_SO2', 'City_CO', 'City_O3']
features = station_feats + city_feats
target = 'AQI'

def load_fused_datasets():
    print("Loading datasets to fuse for maximum accuracy...")
    
    # 1. Load Hourly Station Data
    station_hour_path = ensure_existing_file(DATASET_DIR / "station_hour.csv", "Hourly station dataset")
    stn_hour = pd.read_csv(station_hour_path)
    ensure_columns(stn_hour, ['StationId', 'Datetime', target, *station_feats], station_hour_path.name)
    stn_hour['Datetime'] = pd.to_datetime(stn_hour['Datetime'])
    
    # 2. Load Stations metadata (to get City mappings)
    stations_path = ensure_existing_file(DATASET_DIR / "stations.csv", "Stations metadata")
    stations = pd.read_csv(stations_path)
    ensure_columns(stations, ['StationId', 'City'], stations_path.name)
    
    # 3. Load Hourly City Data
    city_hour_path = ensure_existing_file(DATASET_DIR / "city_hour.csv", "Hourly city dataset")
    city_hour = pd.read_csv(city_hour_path)
    ensure_columns(city_hour, ['City', 'Datetime', *station_feats], city_hour_path.name)
    city_hour['Datetime'] = pd.to_datetime(city_hour['Datetime'])
    
    # 4. Merge Station Hourly with Station Metadata to get City
    df_merged = pd.merge(stn_hour, stations[['StationId', 'City']], on='StationId', how='left')
    
    # 5. Merge City features as macro-level context (rename to avoid collisions)
    city_hour_renamed = city_hour.rename(columns={
        'PM2.5': 'City_PM2.5', 'PM10': 'City_PM10', 'NO2': 'City_NO2',
        'SO2': 'City_SO2', 'CO': 'City_CO', 'O3': 'City_O3', 'AQI': 'City_AQI'
    })
    
    # Merge on City and Datetime
    df_final = pd.merge(df_merged, city_hour_renamed[['City', 'Datetime', 'City_PM2.5', 'City_PM10', 'City_NO2', 'City_SO2', 'City_CO', 'City_O3']], 
                        on=['City', 'Datetime'], how='left')
    
    # Sort chronologically by StationId
    df_final = df_final.sort_values(by=['StationId', 'Datetime']).reset_index(drop=True)
    
    # Forward and backward fill per station to clean sparse gaps
    print("Imputing missing features globally...")
    df_final[features + [target]] = df_final.groupby('StationId')[features + [target]].ffill().bfill()
    
    # Drop rows that still have NaNs (e.g. absolutely no data for that station)
    df_final = df_final.dropna(subset=features + [target])
    
    print(f"Data fused and preprocessed! Final massive shape: {df_final.shape}")
    return df_final

def create_sequences(data, scaler_X, scaler_y, look_back=24):
    scaled_features = scaler_X.transform(data[features])
    scaled_target = scaler_y.transform(data[[target]])

    X, y = [], []
    for i in range(len(scaled_features) - look_back):
        X.append(scaled_features[i:(i + look_back)])
        y.append(scaled_target[i + look_back])
        
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    dense_features = Dense(32, activation='relu', name='feature_dense')(x)
    outputs = Dense(1, activation='linear')(dense_features)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    df = load_fused_datasets()
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaler_X.fit(df[features])
    scaler_y.fit(df[[target]])
    
    joblib.dump(scaler_X, PROJECT_ROOT / 'scaler_X_adv.pkl')
    joblib.dump(scaler_y, PROJECT_ROOT / 'scaler_y_adv.pkl')
    
    look_back = 24  # Look back 24 hours (1 full day)
    
    # Limit to a subset of data to avoid taking days to train on CPU.
    X_all, y_all = [], []
    station_counts = df['StationId'].value_counts()
    eligible_stations = station_counts[station_counts > look_back].index.to_numpy()
    if len(eligible_stations) == 0:
        raise ValueError("No stations have enough hourly history to build 24-step sequences.")
    
    # To save time and RAM, let's randomly sample 15 stations to train on
    np.random.seed(42)
    sample_stations = np.random.choice(
        eligible_stations,
        size=min(15, len(eligible_stations)),
        replace=False,
    )
    
    print(f"Creating sequences mapped on 24-hour cycles for {len(sample_stations)} sample stations...")
    for station in sample_stations:
        group = df[df['StationId'] == station]
        if len(group) > look_back:
            X_station, y_station = create_sequences(group, scaler_X, scaler_y, look_back)
            X_all.append(X_station)
            y_all.append(y_station)
            
    X, y = stack_sequence_groups(X_all, y_all, "hourly fused station data")
    
    print(f"Total 24H sequences ready training: {X.shape[0]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    lstm_model = build_lstm_model((look_back, len(features)))
    
    print("Training Advanced LSTM model (More Neurons, Deeper Lookback)...")
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
    lstm_model.save(PROJECT_ROOT / "lstm_advanced_model.h5")
    
    print("Extracting Deep features...")
    feature_extractor = Model(inputs=lstm_model.inputs, outputs=lstm_model.get_layer('feature_dense').output)
    
    lstm_features_train = feature_extractor.predict(X_train)
    lstm_features_test = feature_extractor.predict(X_test)
    
    print("Training Tuned XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(lstm_features_train, y_train.ravel())
    xgb_model.save_model(PROJECT_ROOT / "xgboost_advanced.json")
    
    print("Evaluating Advanced Hybrid Model...")
    y_pred_scaled = xgb_model.predict(lstm_features_test)
    
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_unscaled = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)
    
    print("-" * 30)
    print("Advanced Model Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
