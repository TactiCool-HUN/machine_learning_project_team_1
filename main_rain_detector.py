from analyze_data_fillna_0 import analyze
from scenario_detector import find_rain_scenarios, extract_week


import downsampler as ds
# 1. Load hourly data for a street
df_raw = ds.get_hourly('Saaritie', lht_included=True)

# 2. Clean missing values (analyze() must return df_no_missing)
df_clean = analyze(df_raw)

# 3. Detect 3 rainfall scenarios
scenarios = find_rain_scenarios(df_clean)
print("Detected scenarios:", scenarios)

# 4. Extract 3 weeks
week_no_rain = extract_week(df_clean, scenarios["NO_RAIN_WEEK"])
week_small    = extract_week(df_clean, scenarios["MANY_SMALL_RAIN_WEEK"])
week_heavy    = extract_week(df_clean, scenarios["HEAVY_RAIN_WEEK"])

# Now you have:
# week_no_rain  → DataFrame for no-rain scenario
# week_small    → DataFrame for many-small-rains
# week_heavy    → DataFrame for heavy-rain

print("No rain week shape:", None if week_no_rain is None else week_no_rain.shape)
print("Many small rains week shape:", None if week_small is None else week_small.shape)
print("Heavy rain week shape:", None if week_heavy is None else week_heavy.shape)