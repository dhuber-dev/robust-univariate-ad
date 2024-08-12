import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
import math
from tqdm import tqdm
import argparse


def get_time_series(data_path: str | float) -> list | None:
    """Load time series data from a given path.

    :param data_path: The path to the CSV file containing time series data or NaN.
    :type data_path: str or float

    :returns: A list of time series values or None if the path is NaN.
    :rtype: list or None
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


def downsample(group, interval: int) -> pd.DataFrame:
    """Downsample the time series data by a given interval.

    :param group: A DataFrame group representing a single time series.
    :type group: pd.DataFrame
    :param interval: The interval at which to downsample the data.
    :type interval: int

    :returns: A downsampled DataFrame.
    :rtype: pd.DataFrame
    """
    group = group.drop(columns=['series_id', 'algo_family_id'])

    downsampled_group = group.groupby(group['time_step'] // interval).agg({
        'test_data': 'mean',
        'time_step': 'first'
    })

    downsampled_group['algo_family_id'] = group['algo_family_id'].iloc[0]
    downsampled_group['series_id'] = group['series_id'].iloc[0]

    return downsampled_group


def load_and_preprocess_data(tsad_results_path: str, time_series_metadata_path: str) -> pd.DataFrame:
    """Load and preprocess the data from specified file paths.

    :param tsad_results_path: Path to the TSAD evaluation results CSV file.
    :type tsad_results_path: str
    :param time_series_metadata_path: Path to the time series metadata CSV file.
    :type time_series_metadata_path: str

    :returns: A DataFrame ready for downsampling and feature extraction.
    :rtype: pd.DataFrame
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

    tqdm.pandas()
    df['test_data'] = df['test_path'].progress_apply(get_time_series).values

    if df.test_data.isnull().any():
        raise ValueError('Could not load data for all instances.')

    algo_family_ids = pd.factorize(df['algo_family'])
    time_steps = df['test_data'].apply(lambda x: list(range(len(x))))

    df_feature_extraction = pd.DataFrame({
        'algo_family_id': algo_family_ids[0],
        'test_data': df['test_data'],
        'time_step': time_steps
    })

    df_feature_extraction = df_feature_extraction.explode(['test_data', 'time_step'])
    df_feature_extraction[['test_data', 'time_step']] = df_feature_extraction[['test_data', 'time_step']].apply(
        pd.to_numeric, errors='coerce')

    return df_feature_extraction


def main(tsad_results_path: str, time_series_metadata_path: str, downsampling_interval: int, output_path: str) -> None:
    """Main function to load data, downsample, and extract features.

    :param tsad_results_path: Path to the TSAD evaluation results CSV file.
    :type tsad_results_path: str
    :param time_series_metadata_path: Path to the time series metadata CSV file.
    :type time_series_metadata_path: str
    :param downsampling_interval: Interval for downsampling the time series.
    :type downsampling_interval: int
    :param output_path: Path to save the extracted features CSV file.
    :type output_path: str

    :returns: None. The extracted features are saved to the specified output path.
    """
    df_feature_extraction = load_and_preprocess_data(tsad_results_path, time_series_metadata_path)

    df_feature_extraction['series_id'] = (df_feature_extraction['time_step'] == 0).cumsum()

    df_downsampled = df_feature_extraction.groupby(['series_id']).apply(downsample,
                                                                        interval=downsampling_interval).reset_index(
        drop=True)

    extracted_features = extract_features(df_downsampled, column_id='algo_family_id', column_sort='time_step')
    extracted_features.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample time series and extract features using tsfresh.")
    parser.add_argument("--tsad-results", type=str, default='datasets/tsad_evaluation_results_preprocessed.csv',
                        help="Path to the TSAD evaluation results CSV file.")
    parser.add_argument("--time-series-metadata", type=str, default='datasets/GutenTAG/datasets.csv',
                        help="Path to the time series metadata CSV file.")
    parser.add_argument("--downsampling-interval", type=int, default=5,
                        help="Interval for downsampling the time series.")
    parser.add_argument("--output-path", type=str, default='datasets/features.csv',
                        help="Path to save the extracted features CSV file.")

    args = parser.parse_args()

    main(args.tsad_results, args.time_series_metadata, args.downsampling_interval, args.output_path)
