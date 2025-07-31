import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv('Housing.csv')

# Show top 5 rows
print(data.head())

# Check missing values and data types
print(data.info())

# Check for nulls
print(data.isnull().sum())

# Convert categorical columns to numeric using get_dummies
data_encoded = pd.get_dummies(data, drop_first=True)

# Check correlations
plt.figure(figsize=(10,8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Price
sns.histplot(data_encoded['price'], kde=True)
plt.title("Price Distribution")
plt.show()

X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(y_test, y_pred, model_name):
    print(f"----- {model_name} -----")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    print()
    
    return r2  
# Evaluation all models
score_lr = evaluate(y_test, y_pred_lr, "Linear Regression")
score_dt = evaluate(y_test, y_pred_dt, "Decision Tree")
score_rf = evaluate(y_test, y_pred_rf, "Random Forest")

# checking which model is best
best_score = max(score_lr, score_dt, score_rf)
if best_score == score_lr:
    best_model = lr
    model_name = "Linear Regression"
elif best_score == score_dt:
    best_model = dt
    model_name = "Decision Tree"
else:
    best_model = rf
    model_name = "Random Forest"

# using best model
new_prediction = best_model.predict(X_test[:5]) 
print("Best Model:", model_name)
print("Predicted Prices:", new_prediction)

y_pred = rf.predict(X_test)

# first 5 price predictions
print("Predicted Prices:", y_pred[:5])

# actual prices
print("Actual Prices:", y_test[:5].values)

# Sample new house features
new_house = {
    'area': 5000,
    'bedrooms': 3,
    'bathrooms': 2,
    'stories': 2,
    'parking': 1,
    'mainroad_yes': 1,
    'guestroom_yes': 0,
    'basement_yes': 1,
    'hotwaterheating_yes': 0,
    'airconditioning_yes': 1,
    'prefarea_yes': 1,
    'furnishingstatus_semi-furnished': 1,
    'furnishingstatus_unfurnished': 0
}

# Convert to DataFrame
input_df = pd.DataFrame([new_house])

# Predict price
predicted_price = rf.predict(input_df)
print("Predicted House Price:", predicted_price[0])

def predict_price():
    area = int(input("Enter area (in sq ft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    stories = int(input("Enter number of stories: "))
    parking = int(input("Enter number of parking spots: "))
    mainroad = input("Is it on main road? (yes/no): ")
    guestroom = input("Is there a guest room? (yes/no): ")
    basement = input("Is there a basement? (yes/no): ")
    hotwaterheating = input("Hot water heating? (yes/no): ")
    airconditioning = input("Air conditioning available? (yes/no): ")
    prefarea = input("Preferred area? (yes/no): ")
    furnishingstatus = input("Furnishing status? (semi-furnished/unfurnished/furnished): ")

    house = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad_yes': 1 if mainroad == 'yes' else 0,
        'guestroom_yes': 1 if guestroom == 'yes' else 0,
        'basement_yes': 1 if basement == 'yes' else 0,
        'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
        'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
        'prefarea_yes': 1 if prefarea == 'yes' else 0,
        'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
    }

    input_df = pd.DataFrame([house])
    predicted = rf.predict(input_df)
    return predicted[0]

# Call function 
price = predict_price()
print("Predicted House Price:", price)