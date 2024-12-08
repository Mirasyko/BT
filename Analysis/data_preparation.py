import pandas as pd
import os
from typing import List
from functools import reduce
from bs4 import BeautifulSoup


def extract_name(path: str) -> str:
    """
    Extract name from path.

    Example:
        'home/user/data/production/produkce_skola_homolka.csv' -> 'skola_homolka
    """
    return path.split('/')[-1].split('_')[1]+ '_' + path.split('/')[-1].split('_')[-1].split('.')[0]

def read_join(labels_list: List[str], path_list: List[str], path_to_data: str) -> pd.DataFrame:
    dfs = {}

    for path,label in zip(path_list, labels_list):
        df = pd.read_csv(f'{path_to_data}{path}')
        try:
            df = df.drop(columns=['Unnamed: 0'])
        except: # column not present in df
            pass

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.columns = [f'{col}_{label}' if col != 'timestamp' else col for col in df.columns]
        dfs[label] = df

    df = reduce(lambda left, right: pd.merge(left, right, on='timestamp'), dfs.values())
    df = df.drop_duplicates(subset='timestamp')
    return df

def create_production_frame(save: bool, path_to_data: str, output_dir: str) -> pd.DataFrame:
    # 1. get list of files and download dataframes
    prod_paths = [f for f in os.listdir(f'{path_to_data}') if 'prod' in f]
    prod_labels = [extract_name(col) for col in prod_paths]

    # 2. join dataframes
    production = read_join(prod_labels, prod_paths, path_to_data)

    if save:
    # 3. save production data
        production.to_csv(f'{output_dir}production.csv', index=False)
    return production

def merge_production_consumption(data_dir: str, output_dir: str):
    production = create_production_frame(save=True, path_to_data=data_dir, output_dir=output_dir)
    # 4. read, transform, save data for consumption
    cons = pd.read_csv(f'{data_dir}/beroun_consumptions.csv').drop(columns=['Unnamed: 0'])
    cons['timestamp'] = pd.to_datetime(cons['timestamp'])
    cons.columns = [f'cons_{col}' if col != 'timestamp' else col for col in cons.columns]
    cons.to_csv(f'{output_dir}consumption.csv', index=False)

    # 5. join prod and cons and save
    joined = pd.merge(cons, production, on='timestamp')
    joined = joined.drop_duplicates(subset='timestamp')
    joined['Date'] = joined['timestamp'].dt.date
    joined.to_csv(f'{output_dir}production_consumption.csv', index=False)
    return

def weather_data(data_dir: str, ref_date: int, output_dir: str) -> pd.DataFrame:
    """
    1. Gets paths of preprocessed weather data in specific format.
    2. Creates date from columns and truncates it with respect to reference date, transforms date to datetime
    3. Join all the weather data and saves it to frame

    Args:
        data_dir (str): path to weather data

    Returns:
        pd.DataFrame: final joined data frame
    """
    paths = os.listdir(data_dir)
    dfs = []

    for path in paths:
        if path.endswith('.csv'):
            df = pd.read_csv(f'{data_dir}{path}')
            df['Date'] = df['Year'].astype(str) + '/' + df['Month'].astype(str).str.zfill(2) + '/' + df['Day'].astype(str).str.zfill(2)
            trunc_df = df[df['Year'] > ref_date]
            trunc_df['Date'] = pd.to_datetime(trunc_df['Date'])
            trunc_df = trunc_df[['Date', 'Value']]
            trunc_df = trunc_df.rename(columns={'Value': path.split('_')[0]})
            dfs.append(trunc_df)

    df_final = reduce(lambda left, right: pd.merge(left, right, on='Date'), dfs)
    df_final.to_csv(f'{output_dir}weather.csv', index=False)

    return 


def sunrise_sunset_dataset(path: str):
    html_path = f'{path}SunSetSunRise.html'
    with open(html_path, 'r', encoding='ISO-8859-2') as f:
        html = f.read()
    
    parsed_html = BeautifulSoup(html)
    td_elements = parsed_html.find_all('td', class_='shadow')
    for td in td_elements:
        data = []
        for td in td_elements:
            text = td.text.strip()
            parts = text.split('Východ:')
            if len(parts) > 1:
                day = parts[0].strip()
                times = parts[1].split('Západ:')
                sunrise = times[0].strip()
                sunset = times[1].strip()
                data.append([day, sunrise, sunset])

        df = pd.DataFrame(data, columns=['Day', 'SunRise','SunSet'])
        df['Day'] = df['Day'].apply(lambda x: x.split('.')[1] if '.' in x else x)

    return df

def create_sunrise_sunset_dataset(df: pd.DataFrame, start: str, end: str, output_dir: str):
    date_range = pd.date_range(start=start, end=end)
    df['Date'] = date_range
    df['SunRise'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['SunRise'], format='%Y-%m-%d %H:%M')
    df['SunSet'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['SunSet'], format='%Y-%m-%d %H:%M')
    df['Daylight'] = df['SunSet'] - df['SunRise'] # Compute the daylight duration
    df['DaylightMinutes'] = df['Daylight'].dt.total_seconds() / 60 # Convert daylight duration to minutes
    df = df[['Date', 'SunRise', 'SunSet', 'Daylight', 'DaylightMinutes']]
    df.to_csv(f'{output_dir}SunSetSunRise.csv', index=False)


def merge_all_dataframes(output_dir: str):
    cons_prod = pd.read_csv(f'{output_dir}production_consumption.csv')
    weather = pd.read_csv(f'{output_dir}weather.csv')
    sunrise_sunset = pd.read_csv(f'{output_dir}SunSetSunRise.csv')

    data_frames = [cons_prod, weather, sunrise_sunset]
    df_final = reduce(lambda left, right: pd.merge(left, right, on='Date'), data_frames)
    df_final.to_csv(f'{output_dir}final.csv', index=False)

    return df_final

def main():

    # 1. create and merge production and consumption data
    merge_production_consumption(data_dir='Analysis/data/energetics_data/', output_dir='Analysis/data/outputs/')

    # 2. weather data
    weather_data(data_dir='Analysis/data/weather_data/', ref_date=2022, output_dir='Analysis/data/outputs/')

    # 3. sunrise and sunset data
    sunrise_sunset = sunrise_sunset_dataset(path='Analysis/data/weather_data/')
    create_sunrise_sunset_dataset(sunrise_sunset, start='2023-01-01', end='2023-12-31', output_dir='Analysis/data/outputs/')

    # 4. Merge all the dataframes
    merge_all_dataframes(output_dir='Analysis/data/outputs/')
    
    return 0

if __name__ == '__main__':
    main()