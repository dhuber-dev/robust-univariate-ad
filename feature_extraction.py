import pandas as pd
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from tsfresh import extract_relevant_features
import math
from tqdm import tqdm
import argparse

category_to_extract_features_for = 'anomaly_kind'  # "algo_family" or "anomaly_kind"


def get_time_series(data_path):
    """Load time series data from a given path."""
    if isinstance(data_path, str):
        df_ts = pd.read_csv('datasets/GutenTAG/' + data_path, usecols=['value-0'])['value-0']
        return df_ts.tolist()
    elif isinstance(data_path, float):
        if math.isnan(data_path):
            return None
        else:
            raise ValueError("Invalid data path type.")
    else:
        raise TypeError("data_path must be a string or NaN.")


def downsample(group, interval):
    """Downsample the time series data by a given interval."""
    return group.groupby(group['time_step'] // interval).agg({
        'test_data': 'mean',
        'time_step': 'first',
        'series_id': 'first',
        category_to_extract_features_for + '_id': 'first'
    })


def reduce_sample_size(df_samples: pd.DataFrame, num_samples_to_keep: int):
    """Reduces the number of samples in a DataFrame for each algorithm family."""
    df_samples['series_id_change'] = df_samples.groupby(category_to_extract_features_for + '_id')['series_id'].transform(
        lambda x: x != x.shift())
    df_samples['sequential_series_id'] = df_samples.groupby(category_to_extract_features_for + '_id')['series_id_change'].cumsum()
    df_samples.drop(columns='series_id_change', inplace=True)

    if (df_samples.groupby(category_to_extract_features_for + '_id')['sequential_series_id'].max() < num_samples_to_keep).any():
        raise ValueError(
            fr'The data does not contain enough samples for all algorithm families to reduce the sample size to {num_samples_to_keep} per family. Choose a smaller number of samples to keep.')
    else:
        return df_samples.loc[df_samples.sequential_series_id <= num_samples_to_keep]


def load_and_preprocess_data(tsad_results_path, time_series_metadata_path):
    """Load and preprocess the data from specified file paths."""
    eval_results = pd.read_csv(tsad_results_path)
    time_series_data = pd.read_csv(time_series_metadata_path)

    eval_results['dataset_name'] = eval_results['dataset'] + '.' + eval_results['dataset_training_type'].apply(
        str.lower)
    data_paths = time_series_data.set_index('dataset_name').loc[:, 'test_path']
    eval_results_agg = eval_results.join(data_paths, on='dataset_name')

    is_correct_collection = eval_results_agg.collection == 'GutenTAG'
    is_unique_anomaly = eval_results_agg['unique_anomaly_type']
    is_unsupervised = eval_results_agg.dataset_training_type == 'UNSUPERVISED'

    df = eval_results_agg.loc[is_correct_collection & is_unique_anomaly & is_unsupervised]

    if is_unique_anomaly.any():
        df.loc[:, 'anomaly_kind'] = df['anomaly_kind'].apply(lambda x: eval(x)[0])  # Remove list type of column "anomaly_kind"

    tqdm.pandas(desc="Loading time series data")
    df_test_data = df.copy()
    df_test_data.loc[:, 'test_data'] = df_test_data['test_path'].progress_apply(get_time_series).values

    if df_test_data.test_data.isnull().any():
        raise ValueError('Could not load data for all instances.')


    feature_category_ids = pd.factorize(df_test_data[category_to_extract_features_for])
    print('Created indices for categories:\n', pd.DataFrame(feature_category_ids[1], columns=['category_name']))
    time_steps = df_test_data['test_data'].apply(lambda x: list(range(len(x))))

    df_feature_extraction = pd.DataFrame({
        category_to_extract_features_for + '_id': feature_category_ids[0],
        'test_data': df_test_data['test_data'],
        'time_step': time_steps
    })

    df_feature_extraction = df_feature_extraction.explode(['test_data', 'time_step'])
    df_feature_extraction[['test_data', 'time_step']] = df_feature_extraction[['test_data', 'time_step']].apply(
        pd.to_numeric, errors='coerce')
    df_feature_extraction['series_id'] = (df_feature_extraction['time_step'] == 0).cumsum()

    return df_feature_extraction.reset_index(drop=True)


def remove_expensive_features(fc_parameters: dict | ComprehensiveFCParameters) -> dict:
    """Create and return feature extraction parameters with computationally expensive features removed."""
    len_original = len(fc_parameters)
    features_to_remove = [
        'sample_entropy',
        'friedrich_coefficients',
        'approximate_entropy'
    ]
    for feature in features_to_remove:
        fc_parameters.pop(feature, None)

    if len_original == len(fc_parameters):
        raise ValueError("No features were removed although function was called. Check list of features to remove.")

    print(f"{len(features_to_remove)} removed from features.")

    return fc_parameters


# Usage in extract_and_save_features function
def extract_and_save_features(df_feature_extraction: pd.DataFrame, n_jobs: int, limit_features: bool, output_path: str):
    """Extract features from the time series data and save to a CSV file."""
    tqdm.pandas(desc="Extracting features")

    # Get the feature extraction parameters
    fc_parameters = ComprehensiveFCParameters()

    if limit_features:
        fc_parameters = remove_expensive_features(fc_parameters)

    extracted_features = extract_features(df_feature_extraction,
                                          column_id=category_to_extract_features_for + '_id',
                                          column_sort='time_step',
                                          default_fc_parameters=fc_parameters,
                                          n_jobs=n_jobs)
    extracted_features.to_csv(output_path)


def main(tsad_results_path,
         time_series_metadata_path,
         downsampling_interval,
         reduced_sample_size,
         n_jobs,
         limit_features,
         output_path):
    """Main function to load data, downsample, and extract features."""
    df_feature_extraction = load_and_preprocess_data(tsad_results_path, time_series_metadata_path)

    tqdm.pandas(desc="Downsampling time series data")

    if downsampling_interval != 0:
        df_feature_extraction = df_feature_extraction.groupby(['series_id']).progress_apply(downsample,
                                                                                            interval=downsampling_interval).reset_index(
            drop=True)

    if reduced_sample_size != 0:
        df_feature_extraction = reduce_sample_size(df_feature_extraction, reduced_sample_size)
        df_feature_extraction = df_feature_extraction.drop(['series_id', 'sequential_series_id'], axis=1)
    else:
        df_feature_extraction = df_feature_extraction.drop(['series_id'], axis=1)

    extract_and_save_features(df_feature_extraction, n_jobs, limit_features, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample time series and extract features using tsfresh.")
    parser.add_argument("--tsad-results", type=str, default='datasets/tsad_evaluation_results_preprocessed.csv',
                        help="Path to the TSAD evaluation results CSV file.")
    parser.add_argument("--time-series-metadata", type=str, default='datasets/GutenTAG/datasets.csv',
                        help="Path to the time series metadata CSV file.")
    parser.add_argument("--downsampling-interval", type=int, default=0,
                        help="Interval for downsampling the time series.")
    parser.add_argument("--reduced-sample-size", type=int, default=0,
                        help="Number of samples to keep for each algorithm family.")
    parser.add_argument("--n-jobs", type=int, default=0,
                        help="Number of cores to use.")
    parser.add_argument("--limit-features", type=int, default=False,
                        help="Remove computationally expensive features to boost extraction/ if resources limited.")
    parser.add_argument("--output-path", type=str, default='datasets/features.csv',
                        help="Path to save the extracted features CSV file.")

    args = parser.parse_args()

    main(args.tsad_results, args.time_series_metadata, args.downsampling_interval, args.reduced_sample_size,
         args.n_jobs, args.limit_features, args.output_path)
