import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
from downsampler import get_hourly

df = get_hourly('Kotaniementie', lht_included = True)
df = df.set_index("Timestamp")

df["rain"] = (df["precipitationIntensity_mm_h"] > 0).astype(int)
df = df.drop(["precipitationIntensity_mm_h"], axis = 1)

# setting up lag
lags = [1, 2, 3, 6, 12, 24, 25, 26, 48, 72, 96, 120]

for lag in lags:
	df[f"precip_quan_lag_{lag}"] = df["precipitationQuantityDiff_mm"].shift(lag)
	df[f"temp_lag_{lag}"] = df["TempC_SHT"].shift(lag)
	df[f"hum_lag_{lag}"] = df["Hum_SHT"].shift(lag)

df["temp_roll_3"]   = df["TempC_SHT"].rolling(3).mean().shift(1)
df["hum_roll_3"]    = df["Hum_SHT"].rolling(3).mean().shift(1)

df = df.dropna()

# train test split
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df.drop(columns=["rain"])
y_train = train_df["rain"]

X_test = test_df.drop(columns=["rain"])
y_test = test_df["rain"]

# make the model
model = LGBMClassifier(
	n_estimators=500,
	learning_rate=0.03,
	max_depth=-1,
	subsample=0.9,
	colsample_bytree=0.9,
	random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1:", round(f1_score(y_test, predictions), 4))
print("\nClassification Report:\n", classification_report(y_test, predictions))

"""
# show importance of variables
importances = model.feature_importances_
feat_names = X_train.columns

plt.figure(figsize=(8, 12))
plt.barh(feat_names, importances)
plt.title("Feature Importance (LightGBM)")
plt.show()
"""
