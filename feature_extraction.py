import pandas as pd
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh import extract_relevant_features
import math
from tqdm import tqdm
import argparse


def get_time_series(data_path):
    """Load time series data from a given path.

    :param data_path: The path to the CSV file containing time series data or NaN.

    :returns: A list of time series values or None if the path is NaN.
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


def downsample(group, interval):
    """Downsample the time series data by a given interval.

    :param group: A DataFrame group representing a single time series.
    :param interval: The interval at which to downsample the data.

    :returns: A downsampled DataFrame.
    """

    return group.groupby(group['time_step'] // interval).agg({
        'test_data': 'mean',
        'time_step': 'first',
        'series_id': 'first',
        'algo_family_id': 'first'
    })


def reduce_sample_size(df_samples: pd.DataFrame, num_samples_to_keep: int):
    """Reduces the number of samples in a DataFrame for each algorithm family, based on sequential series IDs.

    This function creates a sequential series ID within each algorithm family (identified by `algo_family_id`)
    that increments when the `series_id` changes. It then filters the DataFrame to keep only the specified
    number of samples per algorithm family.

    :param df_samples: The DataFrame containing the samples. Must include 'algo_family_id' and 'series_id' columns.
    :param num_samples_to_keep: The number of samples to keep per algorithm family.

    :returns: A DataFrame containing only the reduced number of samples for each algorithm family.

    :raises ValueError: If any algorithm family does not have enough samples to meet the `num_samples_to_keep` requirement.
    """
    # Creating incrementing `series_id` (counter) per `algo_family_id`
    df_samples['series_id_change'] = df_samples.groupby('algo_family_id')['series_id'].transform(
        lambda x: x != x.shift())  # Identify change in `algo_family`

    # Create the new column with sequential integers per `algo_family_id` that increment only when `series_id` changes
    df_samples['sequential_series_id'] = df_samples.groupby('algo_family_id')['series_id_change'].cumsum()
    df_samples.drop(columns='series_id_change', inplace=True)

    # Only keep the desired amount of samples
    if (df_samples.groupby('algo_family_id')['sequential_series_id'].max() < num_samples_to_keep).any():
        # Does not contain enough samples for all `algo_family_id`
        raise ValueError(
            fr'The data does not contain enough samples for all algorithm families to reduce the sample size to {num_samples_to_keep} per family. Choose a smaller number of samples to keep.')
    else:
        return df_samples.loc[df_samples.sequential_series_id <= num_samples_to_keep]


def load_and_preprocess_data(tsad_results_path, time_series_metadata_path):
    """Load and preprocess the data from specified file paths.

    :param tsad_results_path: Path to the TSAD evaluation results CSV file.
    :param time_series_metadata_path: Path to the time series metadata CSV file.

    :returns: A DataFrame ready for downsampling and feature extraction.
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

    tqdm.pandas(desc="Loading time series data")
    df_test_data = df.copy()
    df_test_data.loc[:, 'test_data'] = df_test_data['test_path'].progress_apply(get_time_series).values

    if df_test_data.test_data.isnull().any():
        raise ValueError('Could not load data for all instances.')

    algo_family_ids = pd.factorize(df_test_data['algo_family'])
    time_steps = df_test_data['test_data'].apply(lambda x: list(range(len(x))))

    df_feature_extraction = pd.DataFrame({
        'algo_family_id': algo_family_ids[0],
        'test_data': df_test_data['test_data'],
        'time_step': time_steps
    })

    df_feature_extraction = df_feature_extraction.explode(['test_data', 'time_step'])
    df_feature_extraction[['test_data', 'time_step']] = df_feature_extraction[['test_data', 'time_step']].apply(
        pd.to_numeric, errors='coerce')
    df_feature_extraction['series_id'] = (df_feature_extraction['time_step'] == 0).cumsum()

    return df_feature_extraction.reset_index(drop=True)


def main(tsad_results_path, time_series_metadata_path, downsampling_interval, reduced_sample_size, n_jobs, output_path):
    """Main function to load data, downsample, and extract features.

    :param tsad_results_path: Path to the TSAD evaluation results CSV file.
    :param time_series_metadata_path: Path to the time series metadata CSV file.
    :param downsampling_interval: Interval for downsampling the time series.
    :param reduced_sample_size: Number of samples to keep for each algorithm family.
    :param output_path: Path to save the extracted features CSV file.

    :returns: None. The extracted features are saved to the specified output path.
    """
    df_feature_extraction = load_and_preprocess_data(tsad_results_path, time_series_metadata_path)

    tqdm.pandas(desc="Downsampling time series data")

    if downsampling_interval != 0:
        df_feature_extraction = df_feature_extraction.groupby(['series_id']).progress_apply(downsample,
                                                                                            interval=downsampling_interval).reset_index(
            drop=True)

    if reduced_sample_size != 0:
        df_feature_extraction = reduce_sample_size(df_feature_extraction, reduced_sample_size)

    df_feature_extraction = df_feature_extraction.drop(['series_id', 'sequential_series_id'], axis=1)
    extracted_features = extract_features(df_feature_extraction,
                                          column_id='algo_family_id',
                                          column_sort='time_step',
                                          n_jobs=n_jobs)
    extracted_features.to_csv(output_path)


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
    parser.add_argument("--output-path", type=str, default='datasets/features.csv',
                        help="Path to save the extracted features CSV file.")

    args = parser.parse_args()

    main(args.tsad_results, args.time_series_metadata, args.downsampling_interval, args.reduced_sample_size,
         args.n_jobs, args.output_path)
