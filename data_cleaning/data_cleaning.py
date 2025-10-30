import pandas as pd
import numpy as np


LHT_files = [
	'data/JKL LHT/Data/LHT65005(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65006(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65007(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65008(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65009(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65010(JKL)-TEMP.csv',
	'data/JKL LHT/Data/LHT65013(JKL)-TEMP.csv',
]

kaakkovuorentie = [
	'data/JKL WS100/Data/Kaakkovuorentie/Kaakkovuorentie_202404-202406.csv',
	'data/JKL WS100/Data/Kaakkovuorentie/Kaakkovuorentie_202407-202412.csv',
	'data/JKL WS100/Data/Kaakkovuorentie/Kaakkovuorentie_202501-202506.csv',
	'data/JKL WS100/Data/Kaakkovuorentie/Kaakkovuorentie_202507-202509.csv',
]

kotaniementie = [
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202101-202106.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202107-202112.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202201-202206.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202207-202212.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202301-202306.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202306-202312.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202401-202406.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202406-202412.csv',
	'data/JKL WS100/Data/Kotaniementie/Kotaniementie_202501-202509.csv',
]

saaritie = [
	'',
]

WS100_files = [kaakkovuorentie + kotaniementie]


def get_hourly(street: str, LHT_included: bool = False) -> pd.DataFrame:
	match street:
		case 'kaakkovuorentie':
			pass
		case 'kotaniementie':
			pass


if __name__ == '__main__':
	get_cleaned_data('ws100')
