"""
Time Series Feature Extraction Script

This script processes time series data by downsampling, reducing sample sizes, and extracting features using
the tsfresh library. It is designed for use with time series anomaly detection (TSAD) data and can remove
computationally expensive features to optimize the feature extraction process.

Usage:
    The script is typically run from the command line with arguments specifying the paths to the input data,
    downsampling interval, sample size reduction, number of jobs, feature limiting options, the category for
    feature extraction, and the output path for the extracted features.

Author: Dennis Huber
Date: 2024-09-04
"""

import pandas as pd
import math
from tqdm import tqdm
import argparse
import dask.dataframe as dd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tqdm.dask import TqdmCallback


def get_time_series(data_path):
    """Load time series data from a given path.

    :param data_path: A string representing the path to the time series data file or NaN for missing data.

    :returns: A list of time series values or None if the data path is NaN.
    """
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


def downsample(group, interval, category_to_extract_features_for):
    """Downsample the time series data by a given interval.

    :param group: A DataFrame containing time series data.
    :param interval: An integer specifying the interval for downsampling.
    :param category_to_extract_features_for: A string indicating the category for feature extraction.

    :returns: A downsampled DataFrame.
    """
    return group.groupby(group['time_step'] // interval).agg({
        'test_data': 'mean',
        'time_step': 'first',
        'series_id': 'first',
        category_to_extract_features_for + '_id': 'first'
    })


def reduce_sample_size(df_samples: pd.DataFrame, num_samples_to_keep: int, category_to_extract_features_for: str):
    """Reduces the number of samples in a DataFrame for each algorithm family.

    :param df_samples: A DataFrame containing the samples.
    :param num_samples_to_keep: An integer specifying the number of samples to keep per family.
    :param category_to_extract_features_for: A string indicating the category for feature extraction.

    :returns: A DataFrame with reduced samples.
    """
    df_samples['series_id_change'] = df_samples.groupby(category_to_extract_features_for + '_id')[
        'series_id'].transform(
        lambda x: x != x.shift())
    df_samples['sequential_series_id'] = df_samples.groupby(category_to_extract_features_for + '_id')[
        'series_id_change'].cumsum()
    df_samples.drop(columns='series_id_change', inplace=True)

    if (df_samples.groupby(category_to_extract_features_for + '_id')[
            'sequential_series_id'].max() < num_samples_to_keep).any():
        raise ValueError(
            fr'The data does not contain enough samples for all algorithm families to reduce the sample size to {num_samples_to_keep} per family. Choose a smaller number of samples to keep.')
    else:
        return df_samples.loc[df_samples.sequential_series_id <= num_samples_to_keep]


def load_and_preprocess_data(tsad_results_path, time_series_metadata_path, category_to_extract_features_for, limit_categories):
    """Load and preprocess the data from specified file paths.

    :param tsad_results_path: A string path to the TSAD evaluation results CSV file.
    :param time_series_metadata_path: A string path to the time series metadata CSV file.
    :param category_to_extract_features_for: A string indicating the category for feature extraction.

    :returns: A DataFrame ready for feature extraction.
    """
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
        df.loc[:, 'anomaly_kind'] = df['anomaly_kind'].apply(
            lambda x: eval(x)[0])  # Remove list type of column "anomaly_kind"

    if limit_categories:  # List of categories to limit to is not empty
        is_to_include = df[category_to_extract_features_for].apply(lambda x: x in limit_categories)
        df = df.loc[is_to_include]

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


def remove_expensive_features(fc_parameters: list | ComprehensiveFCParameters) -> list:
    """Create and return feature extraction parameters with computationally expensive features removed.

    :param fc_parameters: A list or ComprehensiveFCParameters object containing feature extraction parameters.

    :returns: A dictionary of feature extraction parameters with expensive features removed.
    """
    len_original = len(fc_parameters)
    features_to_remove = [
        'approximate_entropy'
        'change_quantiles',
        'sample_entropy',
    ]
    for feature in features_to_remove:
        fc_parameters.pop(feature, None)

    if len_original == len(fc_parameters):
        raise ValueError("No features were removed although function was called. Check list of features to remove.")

    print(f"{len(features_to_remove)} removed from features.")

    return fc_parameters


def extract_and_save_features(df_feature_extraction: pd.DataFrame, n_jobs: int, limit_features: str,
                              category_to_extract_features_for: str, output_path: str):
    """Extract features from the time series data using Dask and save to a CSV file.

    :param df_feature_extraction: A DataFrame containing the preprocessed time series data.
    :param n_jobs: An integer specifying the number of jobs (cores) to use for feature extraction.
    :param limit_features: A list indicating set of features to extract.
    :param category_to_extract_features_for: A string indicating the category for feature extraction.
    :param output_path: A string specifying the path to save the extracted features CSV file.
    """
    # Convert the pandas DataFrame to a Dask DataFrame
    ddf = dd.from_pandas(df_feature_extraction, npartitions=n_jobs)

    # Get the feature extraction parameters
    features = ComprehensiveFCParameters()
    limit_features = eval(limit_features)
    if len(limit_features) > 0:
        fc_parameters = {key: None for key in limit_features}
    else:
        fc_parameters = remove_expensive_features(features)

    # Use Dask's parallel computation capabilities
    with TqdmCallback(desc="Extracting features with Dask"):
        extracted_features = extract_features(ddf.compute(),
                                              column_id=category_to_extract_features_for + '_id',
                                              column_sort='time_step',
                                              default_fc_parameters=fc_parameters,
                                              n_jobs=n_jobs)

    # Impute the missing values if necessary
    extracted_features = impute(extracted_features)

    # Save the extracted features to a CSV file
    extracted_features.to_csv(output_path)


def main(tsad_results_path,
         time_series_metadata_path,
         downsampling_interval,
         reduced_sample_size,
         n_jobs,
         limit_features,
         category_to_extract_features_for,
         limit_categories,
         output_path):
    """Main function to load data, downsample, and extract features."""
    df_feature_extraction = load_and_preprocess_data(tsad_results_path, time_series_metadata_path,
                                                     category_to_extract_features_for, limit_categories)

    tqdm.pandas(desc="Downsampling time series data")

    if downsampling_interval != 0:
        df_feature_extraction = df_feature_extraction.groupby(['series_id']).progress_apply(downsample,
                                                                                            interval=downsampling_interval,
                                                                                            category_to_extract_features_for=category_to_extract_features_for).reset_index(
            drop=True)

    if reduced_sample_size != 0:
        df_feature_extraction = reduce_sample_size(df_feature_extraction, reduced_sample_size,
                                                   category_to_extract_features_for)
        df_feature_extraction = df_feature_extraction.drop(['series_id', 'sequential_series_id'], axis=1)
    else:
        df_feature_extraction = df_feature_extraction.drop(['series_id'], axis=1)

    extract_and_save_features(df_feature_extraction, n_jobs, limit_features, category_to_extract_features_for,
                              output_path)


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
    parser.add_argument("--limit-features", type=str, default=[],
                        help="Only use predefined subset of features to boost extraction if resources limited.")
    parser.add_argument("--category", type=str, default="algo_family",
                        help='Category to extract features for ("algo_family" or "anomaly_kind").')
    parser.add_argument("--limit-categories", type=str, default=[],
                        help='Only perform feature extraction on a subset of categories.')
    parser.add_argument("--output-path", type=str, default='datasets/features.csv',
                        help="Path to save the extracted features CSV file.")

    args = parser.parse_args()

    main(args.tsad_results, args.time_series_metadata, args.downsampling_interval, args.reduced_sample_size,
         args.n_jobs, args.limit_features, args.category, args.limit_categories, args.output_path)
