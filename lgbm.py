import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from downsampler import get_hourly

# --- load data ---
df = get_hourly('Kotaniementie', lht_included=True)
df = df.set_index("Timestamp").sort_index()

# --- target: 3 hours ahead ---
df["rain_3h_ahead"] = df["precipitationIntensity_mm_h"].shift(-3) > 0
df["rain_3h_ahead"] = df["rain_3h_ahead"].astype(int)
df = df.drop(columns=["precipitationIntensity_mm_h"])

# --- lag features ---
lags = [1, 2, 3, 6, 12, 24, 48, 72]  # past hours
for lag in lags:
	df[f"precip_quan_lag_{lag}"] = df["precipitationQuantityDiff_mm"].shift(lag)
	df[f"temp_lag_{lag}"] = df["TempC_SHT"].shift(lag)
	df[f"hum_lag_{lag}"] = df["Hum_SHT"].shift(lag)

# --- Rolling features ---
df["temp_roll_3"] = df["TempC_SHT"].rolling(window=3, min_periods=1).mean().shift(1)
df["hum_roll_3"] = df["Hum_SHT"].rolling(window=3, min_periods=1).mean().shift(1)

df = df.dropna()

# --- train/test split (chronological) ---
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns=["rain_3h_ahead"])
y_train = train_df["rain_3h_ahead"]

X_test = test_df.drop(columns=["rain_3h_ahead"])
y_test = test_df["rain_3h_ahead"]

# --- model ---
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

# --- evaluation ---
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# --- optional: baseline for 3h ahead ---
y_pred_baseline = X_test['precip_quan_lag_3'].apply(lambda x: 1 if x > 0 else 0)
print("\n3h-Ahead Baseline Accuracy:", accuracy_score(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_baseline))
