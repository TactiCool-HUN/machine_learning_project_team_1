import pandas as pd
import numpy as np
import os


"""
Recommended functions:
get_hourly()
get_ten_minutely()
"""

_LHT_files = {
	'LHT65005': 'data/JKL LHT/Data/LHT65005(JKL)-TEMP.csv',
	'LHT65006': 'data/JKL LHT/Data/LHT65006(JKL)-TEMP.csv',
	'LHT65007': 'data/JKL LHT/Data/LHT65007(JKL)-TEMP.csv',
	'LHT65008': 'data/JKL LHT/Data/LHT65008(JKL)-TEMP.csv',
	'LHT65009': 'data/JKL LHT/Data/LHT65009(JKL)-TEMP.csv',
	'LHT65010': 'data/JKL LHT/Data/LHT65010(JKL)-TEMP.csv',
	'LHT65013': 'data/JKL LHT/Data/LHT65013(JKL)-TEMP.csv',
}

_street_path = 'data/JKL WS100/Data/'


def _haversine(lat1, lon1, lat2, lon2):
	earth_radius_km = 6371
	lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
	delta_lat = lat2 - lat1
	delta_lon = lon2 - lon1
	a = np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
	return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def _fix_gaps(df: pd.DataFrame) -> pd.DataFrame:
	df_fixed: pd.DataFrame = df.copy()
	if 'Timestamp' in df_fixed.columns:
		timestamp = df_fixed['Timestamp']
		df_fixed = df_fixed.drop(columns=['Timestamp'])
		df_fixed = df_fixed.interpolate(method='linear', limit_direction='both')
		df_fixed.insert(0, 'Timestamp', timestamp)
	else:
		df_fixed = df_fixed.interpolate(method='linear', limit_direction='both')
	return df_fixed


def _get_closest_lht(street: str, resample_frequency: str, based_on_closest_n: int = 3) -> pd.DataFrame:
	street_locations = pd.read_csv("data/JKL WS100/Data/JKL_WS100_sensor_locations.csv", sep = ';')
	street_locations = street_locations[street_locations['Name'] == street]
	street_locations = {
		"latitude": street_locations.Latitude,
		"longitude": street_locations.Longitude,
	}

	lht_locations = pd.read_csv("data/JKL LHT/LHT-SensorLocations.csv")

	lht_locations['distance_km'] = lht_locations.apply(
		lambda row: _haversine(street_locations["latitude"], street_locations["longitude"], row.latitude, row.longitude),
		axis = 1
	)
	lht_locations = lht_locations.loc[lht_locations.nsmallest(based_on_closest_n, 'distance_km').index]
	lht_locations: pd.DataFrame = lht_locations.drop(columns = ['latitude', 'longitude'])

	dfs = []
	for _, location in lht_locations.iterrows():
		sensor_name = location['Temperature sensor']
		df = pd.read_csv(_LHT_files[sensor_name], sep= ';')
		df['Timestamp'] = pd.to_datetime(df['Timestamp'])
		df = df.set_index('Timestamp')
		df = df.resample(resample_frequency).mean().reset_index()
		df = _fix_gaps(df)
		dfs.append(df)

	combined = pd.concat(dfs).groupby('Timestamp', as_index=False).mean()
	
	return combined


def get_hourly(street: str, **kwargs) -> pd.DataFrame:
	"""
	Standardizes timestamps into a nice looking 1-hour interval
	:param street: 
	:key lht_included: whether include lht or not
	:key year_average: will only return 1 year of data, which is averaged out of all years
	:return: 
	"""
	df = _get_standardized_time(street, kwargs.get('lht_included', False), 'h')
	if kwargs.get('year_average', False):
		df = _averaged_year(df)
	return df


def get_ten_minutely(street: str, **kwargs) -> pd.DataFrame:
	"""
	Standardizes timestamps into a nice looking 10-minute interval
	:param street: 
	:key lht_included: whether include lht or not
	:key year_average: will only return 1 year of data, which is averaged out of all years
	:return: 
	"""
	df = _get_standardized_time(street, kwargs.get('lht_included', False), '10min')
	if kwargs.get('year_average', False):
		df = _averaged_year(df)
	return df


def _get_standardized_time(street: str, lht_included: bool = False, resample_frequency: str = '10min') -> pd.DataFrame:
	path = _street_path + street.capitalize() + "/"
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
		df = df.drop(['precipitationIntensity_mm_min', 'precipitationQuantityAbs_mm'], axis = 1)
		# remains:
		# precipitationIntensity_mm_h
		# precipitationQuantityDiff_mm
		# precipitationType
		df.fillna(0, inplace=True)

		def mode_or_nan(x):
			return x.mode().iloc[0] if not x.mode().empty else None

		df_standardized = df.resample(resample_frequency).agg(
			{col: 'mean' for col in df.columns if col != 'precipitationType'} | {'precipitationType': mode_or_nan}
		).reset_index()

		if lht_included:
			df = _get_closest_lht(street, resample_frequency)
			df['Timestamp'] = pd.to_datetime(df['Timestamp'])
			df = df.set_index('Timestamp')
			df = df.resample(resample_frequency).mean().reset_index()
			df_standardized = pd.merge(df_standardized, df, on= 'Timestamp', how= 'left')
		
		dfs.append(df_standardized)

	df = pd.concat(dfs, ignore_index=True)
	return df


def _averaged_year(df: pd.DataFrame) -> pd.DataFrame:
	raise NotImplementedError


if __name__ == '__main__':
	df_out = get_hourly('Saaritie', lht_included = True)

	print(df_out.head())
