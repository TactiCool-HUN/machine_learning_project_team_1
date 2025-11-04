import pandas as pd
import numpy as np
import os


LHT_files = {
	'LHT65005': 'data/JKL LHT/Data/LHT65005(JKL)-TEMP.csv',
	'LHT65006': 'data/JKL LHT/Data/LHT65006(JKL)-TEMP.csv',
	'LHT65007': 'data/JKL LHT/Data/LHT65007(JKL)-TEMP.csv',
	'LHT65008': 'data/JKL LHT/Data/LHT65008(JKL)-TEMP.csv',
	'LHT65009': 'data/JKL LHT/Data/LHT65009(JKL)-TEMP.csv',
	'LHT65010': 'data/JKL LHT/Data/LHT65010(JKL)-TEMP.csv',
	'LHT65013': 'data/JKL LHT/Data/LHT65013(JKL)-TEMP.csv',
}

street_path = 'data/JKL WS100/Data/'


def haversine(lat1, lon1, lat2, lon2):
	R = 6371  # Earth radius in km
	lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
	d_lat = lat2 - lat1
	d_lon = lon2 - lon1
	a = np.sin(d_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(d_lon/2)**2
	return 2 * R * np.arcsin(np.sqrt(a))


def get_closest_lht(street: str, based_on_closest_n: int = 3) -> pd.DataFrame:
	street_locations = pd.read_csv("data/JKL WS100/Data/JKL_WS100_sensor_locations.csv", sep = ';')
	street_locations = street_locations[street_locations['Name'] == street]
	street_locations = {
		"latitude": street_locations.Latitude,
		"longitude": street_locations.Longitude,
	}

	lht_locations = pd.read_csv("data/JKL LHT/LHT-SensorLocations.csv")

	lht_locations['distance_km'] = lht_locations.apply(
		lambda row: haversine(street_locations["latitude"], street_locations["longitude"], row.latitude, row.longitude),
		axis = 1
	)
	lht_locations = lht_locations.loc[lht_locations.nsmallest(based_on_closest_n, 'distance_km').index]
	lht_locations: pd.DataFrame = lht_locations.drop(columns = ['latitude', 'longitude'])

	dfs = []
	max_gap_hours = 2
	
	for _, location in lht_locations.iterrows():
		sensor_name = location['Temperature sensor']
		df = pd.read_csv(LHT_files[sensor_name], sep=';')
		df['Timestamp'] = pd.to_datetime(df['Timestamp'])
		df = df.set_index('Timestamp')
		df = df.resample('h').mean()
	
		# --- FIXED gap length calculation ---
		mask_nan = df.isna()
		gap_lengths = pd.DataFrame(index=df.index, columns=df.columns)
	
		for col in df.columns:
			groups = mask_nan[col].ne(mask_nan[col].shift()).cumsum()
			gap_lengths[col] = mask_nan[col].astype(int).groupby(groups).transform('size')
	
		short_gaps = (mask_nan & (gap_lengths <= max_gap_hours))
	
		df_interp = df.interpolate(method='time', limit=max_gap_hours, limit_direction='both')
		df[df.columns] = np.where(short_gaps, df_interp, df)
		# ------------------------------------
	
		df.columns = pd.MultiIndex.from_product([[sensor_name], df.columns])
		dfs.append(df)
	
	combined = pd.concat(dfs, axis=1)
	mean_df = combined.groupby(level=1, axis=1).mean()
	
	return mean_df


def get_hourly(street: str, lht_included: bool = False) -> pd.DataFrame:
	path = street_path + street.capitalize() + "/"
	csv_files = [
		os.path.join(path, f)
		for f in os.listdir(path)
		if f.endswith(".csv")
	]
	
	dfs = []
	for f in csv_files:
		df = pd.read_csv(f, sep = ';')
		df['Timestamp'] = pd.to_datetime(df['Timestamp'])
		df = df.set_index('Timestamp')

		def mode_or_nan(x):
			return x.mode().iloc[0] if not x.mode().empty else None

		df_hourly = df.resample('h').agg(
			{col: 'mean' for col in df.columns if col != 'precipitationType'} | {'precipitationType': mode_or_nan}
		).reset_index()

		if lht_included:
			df = get_closest_lht(street)
			df['Timestamp'] = pd.to_datetime(df['Timestamp'])
			df = df.set_index('Timestamp')
			df = df.resample('h').mean().reset_index()
			df_hourly = pd.merge(df_hourly, df, on='Timestamp', how='left')
		
		dfs.append(df_hourly)

	df = pd.concat(dfs, ignore_index=True)
	return df


if __name__ == '__main__':
	df_out = get_hourly('Saaritie', True)

	print(df_out.head())
	pass
