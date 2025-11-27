from analyze_data_fillna_0 import analyze
from scenario_detector import find_rain_scenarios, extract_week
import downsampler as ds


def get_scenarios(prints: bool = False):
	# 1. Load hourly data for a street
	df_raw = ds.get_hourly('Saaritie', lht_included=True)
	
	# 2. Clean missing values (analyze() must return df_no_missing)
	df_clean = analyze(df_raw)
	
	# 3. Detect 3 rainfall scenarios
	scenarios = find_rain_scenarios(df_clean)
	if prints:
		print("Detected scenarios:", scenarios)
	
	# 4. Extract 3 weeks
	week_no_rain = extract_week(df_clean, scenarios["NO_RAIN_WEEK"])
	week_small    = extract_week(df_clean, scenarios["MANY_SMALL_RAIN_WEEK"])
	week_heavy    = extract_week(df_clean, scenarios["HEAVY_RAIN_WEEK"])
	
	return {
		'dry': week_no_rain,
		'drizzle': week_small,
		'downpour': week_heavy,
	}


if __name__ == '__main__':
	scenarios_out = get_scenarios(prints = True)
	for key in scenarios_out:
		print(f"{key} shape:", None if scenarios_out[key] is None else scenarios_out[key].shape)


pass
