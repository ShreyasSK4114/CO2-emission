from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load model
model = joblib.load("co2_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    
    # Create dataframe for prediction
    sample = pd.DataFrame([{
        "Engine Size(L)": float(data["engine_size"]),
        "Cylinders": int(data["cylinders"]),
        "Fuel Consumption Comb (L/100 km)": float(data["fuel_consumption"]),
        "Vehicle Class": data["vehicle_class"]
    }])
    
    prediction = model.predict(sample)[0]
    
    return render_template("index.html", 
                           prediction_text=f"Predicted CO₂ Emissions: {prediction:.2f} g/km")

if __name__ == "__main__":
    app.run(debug=True)
