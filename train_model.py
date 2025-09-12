import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
df = pd.read_csv("cars.csv")

# 2. Features & target
X = df[["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)", "Vehicle Class"]]
y = df["CO2 Emissions(g/km)"]

# 3. Preprocessor (for Vehicle Class)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Vehicle Class"])
    ],
    remainder="passthrough"
)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create Random Forest pipeline
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train model
rf_model.fit(X_train, y_train)

# 7. Save model
joblib.dump(rf_model, "co2_model.pkl")
print("✅ Model trained and saved as co2_model.pkl")
