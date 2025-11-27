import pandas as pd
import downsampler as ds
from analyze_data_fillna_0 import analyze


def find_rain_scenarios(df, prints: bool = False):
	"""
	Detects:
	1) A continuous 7-day week with NO RAIN
	2) A continuous 7-day week with MANY SMALL RAIN EVENTS
	3) A continuous 7-day week with ONE HEAVY RAIN EVENT
	"""

	df = df.copy()
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])
	df = df.set_index('Timestamp')

	rain = df['precipitationIntensity_mm_h']

	# Expected number of rows per week
	# e.g. hourly = 24*7, 10-min = 7*24*6
	expected_rows = df.resample("D").size().median() * 7

	weekly_groups = rain.resample("W")

	week_no_rain = None
	week_many_small = None
	week_heavy = None

	# noinspection PyPep8Naming
	HEAVY_THRESHOLD = 5.0  # mm/h for heavy rain spike
	# noinspection PyPep8Naming
	SMALL_THRESHOLD = 2.0   # upper limit for "small rains"

	for week_end, week_series in weekly_groups:

		# --- 1. Ensure continuous week (no missing timestamps) ---
		week_series_nonan = week_series.dropna()

		if len(week_series_nonan) != expected_rows:
			continue  # skip incomplete or broken weeks

		# Rain events count (0 -> >0 transitions)
		events = ((week_series_nonan > 0) & 
				  (week_series_nonan.shift(1) == 0)).sum()

		max_intensity = week_series_nonan.max()

		# Debug print
		if prints:
			print(week_end, "events:", events, "max:", max_intensity) 
		# --- Scenario 1: No rain for whole week ---
		if (week_series_nonan == 0).all() and week_no_rain is None:
			week_no_rain = week_end

		# --- Scenario 2: Many small rains ---
		if (events >= 20) and (max_intensity < HEAVY_THRESHOLD) and week_many_small is None:
			week_many_small = week_end

		# --- Scenario 3: One heavy rain event ---
		if (events == 1) and (max_intensity >= HEAVY_THRESHOLD) and week_heavy is None:
			week_heavy = week_end

		# Stop early if all found
		if week_no_rain and week_many_small and week_heavy:
			break

	return {
		"NO_RAIN_WEEK": week_no_rain,
		"MANY_SMALL_RAIN_WEEK": week_many_small,
		"HEAVY_RAIN_WEEK": week_heavy
	}


def extract_week(df, week_end):
	"""
	Returns the full 7-day window for the scenario.
	"""
	if week_end is None:
		return None

	df = df.copy()
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])

	start = week_end - pd.Timedelta(days=7)
	mask = (df['Timestamp'] >= start) & (df['Timestamp'] <= week_end)

	return df.loc[mask]

if __name__ == "__main__":

	# 1. Load raw data from downsampler
	# List of streets: Kotaniementie, Saaritie, Kaakkovuorentie, Tähtiniementie, Tuulimyllyntie.
	df_raw = ds.get_hourly('Saaritie', lht_included=True)

	# 2. Clean missing values
	df_clean = analyze(df_raw)

	# 3. Run the scenario detector
	scenarios = find_rain_scenarios(df_clean)
	print("Detected scenarios:")
	print(scenarios)

	# 4. Extract weeks
	week1 = extract_week(df_clean, scenarios["NO_RAIN_WEEK"])
	week2 = extract_week(df_clean, scenarios["MANY_SMALL_RAIN_WEEK"])
	week3 = extract_week(df_clean, scenarios["HEAVY_RAIN_WEEK"])

	print("\nExtracted week shapes:")
	print("No rain week:", None if week1 is None else week1.shape)
	print("Many small rains week:", None if week2 is None else week2.shape)
	print("Heavy rain week:", None if week3 is None else week3.shape)

