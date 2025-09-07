
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("Housing dataset.csv")

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]]
y = df["Price"]

# Train Linear Regression
model = LinearRegression()
model.fit(X, y)

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as house_model.pkl")