import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from downsampler import get_hourly

# --- load data ---
df = get_hourly('Kotaniementie', lht_included=True)
df = df.set_index("Timestamp").sort_index()

# --- target ---
df["rain"] = (df["precipitationIntensity_mm_h"] > 0).astype(int)
df = df.drop(columns=["precipitationIntensity_mm_h"])

# --- lag features ---
lags = [1, 2, 3, 6, 12, 24, 48, 72]  # removed risky 25,26,96,120
for lag in lags:
	df[f"precip_quan_lag_{lag}"] = df["precipitationQuantityDiff_mm"].shift(lag)
	df[f"temp_lag_{lag}"] = df["TempC_SHT"].shift(lag)
	df[f"hum_lag_{lag}"] = df["Hum_SHT"].shift(lag)

df["temp_roll_3"] = df["TempC_SHT"].rolling(window=3, min_periods=1).mean().shift(1)
df["hum_roll_3"] = df["Hum_SHT"].rolling(window=3, min_periods=1).mean().shift(1)

# --- drop rows with missing values caused by lag/rolling ---
df = df.dropna()

# --- train/test split ---
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns=["rain"])
y_train = train_df["rain"]

X_test = test_df.drop(columns=["rain"])
y_test = test_df["rain"]

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


# Predict rain if it rained in the last hour
y_pred_baseline = X_test['precip_quan_lag_1'].apply(lambda x: 1 if x > 0 else 0)

from sklearn.metrics import accuracy_score, classification_report
print("Baseline Accuracy:", accuracy_score(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_baseline))