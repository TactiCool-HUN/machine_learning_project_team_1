import pandas as pd
import os


LHT_files = [
	'data/JKL LHT/Data/LHT65005(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65006(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65007(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65008(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65009(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65010(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65013(JKL)-TEMP.csv',
]

street_path = 'data/JKL WS100/Data/'


def get_closest_lht(street: str) -> pd.DataFrame:
	return pd.read_csv(LHT_files[0], sep = ';')  # TODO: function


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
		df_hourly = df.resample('h').mean().reset_index() # TODO: don't mean the type column

		if lht_included:
			df = get_closest_lht(street)
			df['Timestamp'] = pd.to_datetime(df['Timestamp'])
			df = df.set_index('Timestamp')
			df = df.resample('h').mean().reset_index()
			df_hourly = pd.merge(df_hourly, df, on='Timestamp', how='left')
		
		dfs.append(df_hourly)

	df = pd.concat(dfs, ignore_index=True)
	
	print(df.head())

	return df


if __name__ == '__main__':
	get_hourly('Saaritie', True)
