import pandas as pd
import os
from typing import List
from functools import reduce


def extract_name(path: str) -> str:
    return path.split('/')[-1].split('_')[1]+ '_' + path.split('/')[-1].split('_')[-1].split('.')[0]

def read_join(labels_list: List[str], path_list: List[str]) -> pd.DataFrame:
    dfs = {}

    for path,label in zip(path_list, labels_list):
        df = pd.read_csv(f'Energetika/energetics_data/{path}').drop(columns=['Unnamed: 0'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.columns = [f'{col}_{label}' if col != 'timestamp' else col for col in df.columns]
        dfs[label] = df

    df = reduce(lambda left, right: pd.merge(left, right, on='timestamp'), dfs.values())
    df = df.drop_duplicates(subset='timestamp')
    return df

def main():

    # 1. get list of files and download dataframes
    prod_paths = os.listdir('Energetika/energetics_data/')
    prod_labels = [extract_name(col) for col in prod_paths]

    # 2. join dataframes
    production = read_join(prod_labels, prod_paths)

    # 3. save production data
    production.to_csv('Energetika/production.csv', index=False)

    # 4. read, transform, save data for consumption
    cons = pd.read_csv('Energetika/beroun_consumptions.csv').drop(columns=['Unnamed: 0'])
    cons['timestamp'] = pd.to_datetime(cons['timestamp'])
    cons.columns = [f'cons_{col}' if col != 'timestamp' else col for col in cons.columns]
    cons.to_csv('Energetika/consumption.csv', index=False)

    # 5. join prod and cons and save
    joined = pd.merge(cons, production, on='timestamp')
    joined = joined.drop_duplicates(subset='timestamp')
    joined.to_csv('Energetika/prod_cons.csv', index=False)

    return 0

if __name__ == '__main__':
    main()