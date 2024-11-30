import pandas as pd
import os

import tsfresh
from tqdm import tqdm
import math

from tsfresh import extract_features

from FeedForward import FFModel
import torch

def get_blocks_of_anomalies(sample):
    # Identify blocks of anomalies (ones)
    sample['block'] = (sample['is_anomaly'] != sample['is_anomaly'].shift()).cumsum() * sample['is_anomaly']
    
    # Group rows by blocks of 1s
    groups = sample[sample['is_anomaly'] == 1].groupby('block')
    
    # Extract sub-DFs
    sub_dfs = [group for _, group in groups]
    
    return sub_dfs


def get_window_of_anomaly(x, length=1000):
    start_anomaly = x.index.min()
    end_anomaly = x.index.max()
    middle = start_anomaly + (end_anomaly - start_anomaly) / 2
    start = math.ceil(middle - length / 2)
    end = math.floor(middle + length / 2)
    if start < 0:
        start = 0
        end = length

    return start, end


def main(dataset):
    dataset_folder = f'datasets/{dataset}/'
    files_to_load = [dataset_folder + f for f in os.listdir(dataset_folder) if f.endswith("test.csv")]

    tqdm.pandas(desc="Loading datasets")
    df = pd.DataFrame({'path': files_to_load})
    df['data'] = df.path.progress_apply(pd.read_csv)

    features_old = pd.read_csv('current_features.csv', index_col=0)
    fc_parameters = tsfresh.feature_extraction.settings.from_columns(features_old)

    model = FFModel(input_shape=(32, features_old.shape[1]),
                    num_classes=4)
    model.load_state_dict(torch.load("ff_model.pth"))
    model.eval()
    results = []

    for d in df.data:
        anomaly_data = []
        anomalies = get_blocks_of_anomalies(d)
        for idx, anomaly in enumerate(anomalies):
            start, end = get_window_of_anomaly(anomaly)
            value_col = [x for x in d.columns if 'value' in x]
            df4extraction = d.loc[start:end][value_col].squeeze().reset_index(drop=True).reset_index().rename({'index': 'time_idx'}, axis=1)
            df4extraction['id'] = idx
            anomaly_data.append(df4extraction)
        all_df4extraction = pd.concat(anomaly_data)


        features = extract_features(all_df4extraction,
                                            column_id='id',
                                            column_sort='time_idx',
                                            n_jobs=1,
                                            default_fc_parameters=fc_parameters['value'])

        algo_families = []
        for _, ts_features in features.iterrows():
            input_tensor = torch.tensor(ts_features.values, dtype=torch.float32)
            # Assume `input_data` is a PyTorch tensor with the required shape
            with torch.no_grad():  # Disable gradient calculation for faster inference
                algo_family = model(input_tensor)
            algo_families.append(algo_family)
        results.append(algo_families)
    df['results'] = results
    df.to_csv(f'results_algo_families_{dataset}.csv')



if __name__ == '__main__':
    for d in ['KDD-TSAD', 'MGAB', 'NAB', 'NASA-MSL', 'NASA-SMAP', 'NormA']:
        df_d = main(d)
