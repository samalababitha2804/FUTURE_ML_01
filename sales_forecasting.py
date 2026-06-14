import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("Superstore.csv", encoding="latin1")

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Create time features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Select features and target
X = df[['Year', 'Month']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

# Forecast next 12 months
future = pd.DataFrame({
    'Year': [2026]*12,
    'Month': list(range(1, 13))
})

future_sales = model.predict(future)

# Plot forecast
plt.figure(figsize=(8,5))
plt.plot(future['Month'], future_sales, marker='o')
plt.title("Sales Forecast for Next 12 Months")
plt.xlabel("Month")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()