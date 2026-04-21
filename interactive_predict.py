import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd


POLLUTANT_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER_FEATURES = ["temperature", "humidity", "windspeed"]


def get_numeric_input(prompt):
    while True:
        raw_value = input(prompt).strip()

        if not raw_value:
            print("Missing value. Using 0.0")
            return 0.0

        try:
            return float(raw_value)
        except ValueError:
            print("Invalid input. Please enter a number.")


def classify_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def ordered_prompt_features(model_features):
    preferred_order = POLLUTANT_FEATURES + WEATHER_FEATURES
    ordered_features = [feature for feature in preferred_order if feature in model_features]
    ordered_features.extend([feature for feature in model_features if feature not in ordered_features])
    return ordered_features


def build_prediction_sequence(current_values, model_features, look_back):
    repeated_rows = [current_values.copy() for _ in range(look_back)]
    return pd.DataFrame(repeated_rows, columns=model_features)


def main():
    print("====================================")
    print(" Real-Time AQI Predictor")
    print("====================================")
    print("Enter the latest realtime pollutant and weather values.")

    try:
        from predict_hybrid_realtime import RealTimeHybridAQIPredictor

        predictor = RealTimeHybridAQIPredictor()
    except Exception as exc:
        print(f"\n[ERROR] Model loading failed: {exc}")
        print("Please ensure the trained artifacts are present and compatible.")
        return

    print(f"The model will reuse the same values for the last {predictor.look_back} steps.\n")

    model_features = predictor.features
    unsupported_inputs = [feature for feature in WEATHER_FEATURES if feature not in model_features]
    prompt_features = ordered_prompt_features(model_features)
    if unsupported_inputs:
        print(f"Current model features: {', '.join(model_features)}")
        print("Note: the loaded model was not trained with "
              f"{', '.join(unsupported_inputs)}, so those inputs will not affect the prediction.")
        print("Retrain `train_hybrid_model.py` with a dataset that contains those columns to use them.\n")

    input_values = {}
    for feature in prompt_features:
        input_values[feature] = get_numeric_input(f"{feature}: ")

    model_values = {feature: input_values[feature] for feature in model_features}
    prediction_sequence = build_prediction_sequence(
        model_values,
        model_features,
        predictor.look_back,
    )

    print("\nProcessing AQI prediction pipeline...")
    try:
        predicted_aqi = predictor.predict_realtime(prediction_sequence)
        category = classify_aqi(predicted_aqi)

        print("\n====================================")
        print(f"Predicted AQI  : {predicted_aqi:.2f}")
        print(f"AQI Category   : {category}")
        print("====================================")
        print(f"Assumption     : same realtime values reused for {predictor.look_back} recent steps.")
        if unsupported_inputs:
            print(f"Unused Inputs   : {', '.join(unsupported_inputs)}")
        print()
    except Exception as exc:
        print(f"\n[ERROR] Model prediction failed: {exc}")
        print("Please ensure the trained artifacts are present and compatible.")


if __name__ == "__main__":
    main()
