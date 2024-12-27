import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#  Veri setini yükleme
data = pd.read_csv(r"C:/Users/utkus/Desktop/pokemon 2/metadata.csv")


#  Özellikler ve hedef sütunlar

target = "hp"  
features = ["attack", "defense", "special-attack", "special-defense", "speed"]



existing_features = [col for col in features if col in data.columns]
data = data.dropna(subset=existing_features + [target])

X = data[existing_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Model oluşturma
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

#  Ölçümler

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
