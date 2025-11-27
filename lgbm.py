import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from downsampler import get_hourly


def make_model(street_name: str, prints: bool = False):
	# --- load data ---
	df = get_hourly(street_name, lht_included=True)
	df = df.set_index("Timestamp").sort_index()
	
	# --- target: 3 hours ahead ---
	df["rain_3h_ahead"] = df["precipitationIntensity_mm_h"].shift(-3) > 0
	df["rain_3h_ahead"] = df["rain_3h_ahead"].astype(int)
	df = df.drop(columns=["precipitationIntensity_mm_h"])
	
	# --- lag features ---
	lags = [1, 2, 3, 6, 12, 24, 48, 72]
	for lag in lags:
		df[f"precip_quan_lag_{lag}"] = df["precipitationQuantityDiff_mm"].shift(lag)
		df[f"temp_lag_{lag}"] = df["TempC_SHT"].shift(lag)
		df[f"hum_lag_{lag}"] = df["Hum_SHT"].shift(lag)
	
	# --- rolling features ---
	df["temp_roll_3"] = df["TempC_SHT"].rolling(window=3, min_periods=1).mean().shift(3)
	df["hum_roll_3"] = df["Hum_SHT"].rolling(window=3, min_periods=1).mean().shift(3)
	
	df = df.dropna()
	
	# --- train/test split ---
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
		random_state=42,
		verbosity=-1
	)
	
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	
	print(street_name)
	# --- evaluation ---
	print("Accuracy:", accuracy_score(y_test, predictions))
	print(classification_report(y_test, predictions))
	
	# --- baseline for 3h ahead ---
	y_pred_baseline = X_test['precip_quan_lag_3'].apply(lambda x: 1 if x > 0 else 0)
	print("\n3h-Ahead Baseline Accuracy:", accuracy_score(y_test, y_pred_baseline))
	print(classification_report(y_test, y_pred_baseline))


if __name__ == '__main__':
	streets = ['Kaakkovuorentie', 'Kotaniementie', 'Saaritie', 'Tuulimyllyntie', 'Tähtiniementie']
	for street in streets:
		make_model(street, prints = True)

pass
