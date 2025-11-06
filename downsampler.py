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
	earth_radius_km = 6371
	lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
	d_lat = lat2 - lat1
	d_lon = lon2 - lon1
	a = np.sin(d_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(d_lon/2)**2
	return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def fix_gaps(df: pd.DataFrame) -> pd.DataFrame:
	df_fixed: pd.DataFrame = df.copy()
	if 'Timestamp' in df_fixed.columns:
		timestamp = df_fixed['Timestamp']
		df_fixed = df_fixed.drop(columns=['Timestamp'])
		df_fixed = df_fixed.interpolate(method='linear', limit_direction='both')
		df_fixed.insert(0, 'Timestamp', timestamp)
	else:
		df_fixed = df_fixed.interpolate(method='linear', limit_direction='both')
	return df_fixed

""" # turns out ppl are smarter than me :c I'm still proud of it tho' even if it's slow af
def fix_gaps_slow(df: pd.DataFrame) -> pd.DataFrame:
	column_gaps = {}  # column_name: [starting_line_index, starting_value]
	df_reduced: pd.DataFrame = df.copy()
	df_reduced = df_reduced.drop('Timestamp', axis = 1)
	previous_line = None
	
	for line in df_reduced.iterrows():
		for column in line[1].items():
			if pd.isna(column[1]):
				if not column_gaps.get(column[0], False):
					# no registered gap -> add gap start
					# column_name: [starting_line_index, starting_value]
					column_gaps[column[0]] = [previous_line[0], previous_line[1][column[0]]]
			else:
				if column_gaps.get(column[0], False):
					# there is a gap, and it has now ended -> fill in empty spots
					start_column = {
						'index': column_gaps[column[0]][0],
						'val': column_gaps[column[0]][1],
					}
					end_column = {
						'index': line[0],
						'val': column[1],
					}
					gap_len = end_column['index'] - start_column['index']
					gap_size = end_column['val'] - start_column['val']
					step = gap_size / gap_len
					
					last_val = start_column['val']
					for i in range(start_column['index'] + 1, end_column['index']):
						last_val += step 
						df.loc[i, column[0]] = last_val
							
		previous_line = line
	
	return df
"""

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
	for _, location in lht_locations.iterrows():
		sensor_name = location['Temperature sensor']
		df = pd.read_csv(LHT_files[sensor_name], sep=';')
		df['Timestamp'] = pd.to_datetime(df['Timestamp'])
		df = df.set_index('Timestamp')
		df = df.resample('h').mean().reset_index()
		df = fix_gaps(df)
		dfs.append(df)

	combined = pd.concat(dfs).groupby('Timestamp', as_index=False).mean()
	
	return combined


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
