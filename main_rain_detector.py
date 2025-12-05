from scenario_detector import find_rain_scenarios, extract_week
from lgbm import feature_engineering


def get_scenarios(df, prints: bool = False):
	df = df.reset_index()
	scenarios = find_rain_scenarios(df)
	if prints:
		print("Detected scenarios:", scenarios)

	return {
		'dry': extract_week(df, scenarios["NO_RAIN_WEEK"]),
		'drizzle': extract_week(df, scenarios["MANY_SMALL_RAIN_WEEK"]),
		'downpour': extract_week(df, scenarios["HEAVY_RAIN_WEEK"]),
	}


if __name__ == '__main__':
	scenarios_out = get_scenarios(feature_engineering('Saaritie'), prints = True)
	for key in scenarios_out:
		print(f"{key} shape:", None if scenarios_out[key] is None else scenarios_out[key].shape)


pass
