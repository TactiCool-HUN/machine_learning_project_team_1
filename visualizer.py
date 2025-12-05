import matplotlib.pyplot as plt
import numpy as np
import os

from main_rain_detector import get_scenarios
from lgbm import make_model, feature_engineering

OUTPUT_DIR = "scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_scenario(df, model, name, threshold=0.5):
	"""
	df: scenario DataFrame, already feature-engineered and with 'rain_3h_ahead'
	model: trained LGBMClassifier
	"""

	# --- prepare inputs ---
	X = df.drop(columns=["rain_3h_ahead", "precipitationIntensity_mm_h"])
	y_true = df["rain_3h_ahead"]

	# --- model predictions ---
	probs = model.predict_proba(X)[:, 1]
	preds = (probs >= threshold).astype(int)

	# --- correctness indicator ---
	correct = preds == y_true

	# --- plotting ---
	plt.figure(figsize=(14, 6))

	# 1) predicted probability curve
	plt.plot(df.index, probs, label="Predicted Probability", linewidth=2)

	# 2) actual rain (0/1)
	plt.step(df.index, y_true, where="mid", label="Actual (rain_3h_ahead)", linewidth=2)

	# 3) prediction dots (green correct / red wrong)
	plt.scatter(
		df.index,
		preds,
		c=np.where(correct, "green", "red"),
		s=40,
		label="Predicted Class",
		alpha=0.85,
	)

	plt.title(f"Scenario: {name}")
	plt.xlabel("Time")
	plt.ylabel("Rain Forecast")
	plt.ylim(-0.1, 1.1)
	plt.grid(True)
	plt.legend()

	# save PNG
	file_name = os.path.join(OUTPUT_DIR, f"scenario_{name}.png")
	plt.savefig(file_name, dpi=200, bbox_inches="tight")
	print(f"Saved: {file_name}")
	plt.close()


def show_week_prediction():
	df = feature_engineering('Saaritie')

	scenarios = get_scenarios(df)
	for key in scenarios:
		scenarios[key] = scenarios[key].set_index("Timestamp").sort_index()

	model = make_model(df, 'Saaritie')

	for name, scenario_df in scenarios.items():
		plot_scenario(scenario_df, model, name)


if __name__ == '__main__':
	show_week_prediction()
