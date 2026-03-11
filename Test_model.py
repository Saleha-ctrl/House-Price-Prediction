
import joblib

model = joblib.load("House_price_prediction_model.pkl")
scale = joblib.load("scaler.pkl")

new_data = [["-122.25","37.85","52.0","1274.0","235.0","558.0","219.0","5.6431"]]

scaled_data = scale.transform(new_data)

predict = model.predict(scaled_data)

print("Prediction = ",predict)

