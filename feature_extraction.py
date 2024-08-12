import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
import math
from tqdm import tqdm

# Specify your file locations here, if changed
TSAD_EVALUATION_RESULTS = 'datasets/tsad_evaluation_results_preprocessed.csv' # already filtered (see `preprocessing_and_analysis_tsad_evaluation_results.ipynb`)
TIME_SERIES_METADATA = 'datasets/GutenTAG/datasets.csv'

eval_results = pd.read_csv(TSAD_EVALUATION_RESULTS)
time_series_data = pd.read_csv(TIME_SERIES_METADATA)

# Get data paths
eval_results['dataset_name'] = eval_results['dataset'] + '.' + eval_results['dataset_training_type'].apply(str.lower) # recreate `dataset_name` column of 'datasets/GutenTAG/datasets.csv'
data_paths = time_series_data.set_index('dataset_name').loc[:,'test_path'] # select how and what to join from 'datasets/GutenTAG/datasets.csv'. Only paths for data used in testing needed
eval_results_agg = eval_results.join(data_paths, on='dataset_name')

# Filter
is_correct_collection = eval_results_agg.collection == 'GutenTAG'
is_unique_anomaly = eval_results_agg['unique_anomaly_type']
is_unsupervised = eval_results_agg.dataset_training_type == 'UNSUPERVISED'

df = eval_results_agg.loc[is_correct_collection & is_unique_anomaly & is_unsupervised]

# Add data
def get_time_series(data_path: str | float) -> list | None:
    if isinstance(data_path, str):
        df_ts = (pd.read_csv('datasets/GutenTAG/' + data_path, usecols=['value-0'])['value-0'])
        return df_ts.tolist()
    elif isinstance(data_path, float):
        if math.isnan(data_path):
            return None
        else:
            raise ValueError
    else:
        raise TypeError


tqdm.pandas()
df['test_data'] = df['test_path'].progress_apply(get_time_series).values

if df.test_data.isnull().any():
    raise ValueError('Could not load data for all instances.')

# Prepare a df for feature extraction by converting it to the specific format and making everything numeric
algo_family_ids = pd.factorize(df['algo_family']) # Translate algo_family ids from string to integer
print(pd.Series(range(len(algo_family_ids[1])),index=algo_family_ids[1],name='algo_family_ids'))
time_steps = df['test_data'].apply(lambda x: list(range(len(x)))) # Creating a column that matches 'test_data' but with time-steps as lists elements instead of values

df_feature_extraction = pd.DataFrame({
    'algo_family_id': algo_family_ids[0],
    'test_data': df['test_data'],
    'time_step': time_steps
})

df_feature_extraction = df_feature_extraction.explode(['test_data', 'time_step'])
df_feature_extraction[['test_data', 'time_step']] = df_feature_extraction[['test_data', 'time_step']].apply(pd.to_numeric, errors='coerce') # Because it was converted from lists, double check for numeric

# Extract features using tsfreshö
extracted_features = extract_features(df_feature_extraction, column_id='algo_family_id', column_sort='time_step')
extracted_features.to_csv('datasets/features.csv')
