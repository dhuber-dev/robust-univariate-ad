# main_script.py
import argparse
import os
import yaml
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from FeedForward import evaluate_FFModel
from BidirectionalLSTM import BidirectionalLSTM
from XGBoost import XGBoostModel


def read_yaml(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_results(output_path, results):
    results_file = os.path.join(output_path, "performance_results.yaml")
    with open(results_file, 'w') as file:
        yaml.dump(results, file)
    print(f"Results saved to {results_file}")


def preprocess_time_series_data(df):
    """
    Converts time series data from long format to wide format.

    :param df: pandas.DataFrame
        Input time series data.
    :return: pandas.DataFrame
        Wide-format data suitable for model input.
    """
    wide_df = df.pivot(index='id', columns='time_idx', values='value')
    wide_df.columns = [f"Feature_{col}" for col in wide_df.columns]
    wide_df.reset_index(drop=True, inplace=True)
    return wide_df


def markdown_report(class_report):
    # Initialize markdown table
    markdown_report = "## Classification Report\n\n"
    markdown_report += "| Class | Precision | Recall | F1-Score | Support |\n"
    markdown_report += "|-------|-----------|--------|----------|---------|\n"

    # Add metrics for each class
    for label, metrics in class_report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Ignore averages
            markdown_report += f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']:.0f} |\n"

    # Add accuracy row
    accuracy = class_report['accuracy']
    markdown_report += f"| Accuracy | - | - | {accuracy:.4f} | {metrics['support']:.0f} |\n"

    # Add macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        metrics = class_report[avg_type]
        markdown_report += f"| {avg_type.title()} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']:.0f} |\n"

    return markdown_report


def main(features, labels, input_type, output_path, hyperparameters, mapping):
    # Load data
    data = pd.read_csv(features, index_col=0)
    labels = pd.read_csv(labels, index_col=0)
    labels = labels.loc[~labels.invalid_data, 'anomaly'].squeeze()

    # Prepare
    ## Features
    if input_type == "time-series":
        features = preprocess_time_series_data(data)
    elif input_type == 'features':
        features = data.dropna(axis=1)
        features = features.loc[:, features.nunique() > 1] # Remove columns with just one value
        print(len(features.columns), 'features used.')
        features = features.clip(lower=-1e6, upper=1e6)

    else:
        raise ValueError

    ## Labels
    if mapping:
        labels = labels.map(read_yaml('best_performer_mapper.yaml'))
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    ## Split & scale
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_pred = evaluate_FFModel(X_train, y_train, X_test, y_test, hyperparameters)

    print(classification_report(y_test, y_pred, zero_division=0))
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    report = markdown_report(class_report)

    learning_rate, batch_size, num_epochs = hyperparameters
    report += f'\nInput Type: {input_type}\nLearning Rate: {learning_rate}\nBatch Size: {batch_size}\nNumber of Epochs: {num_epochs}'
    report += f'\nMapping: {[f'- {l}: {l_int}' for l, l_int in label_mapping.items()]}'

    # Save the markdown report to a .md file
    with open(output_path, "w") as f:
        f.write(report)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a model for time series data.")
    parser.add_argument("--labels", "-l",
                        default='loaded_self_generated_df.csv',
                        type=str,
                        help="Path to the label CSV file.")
    parser.add_argument("--features", "-f",
                        default='exploded_self_generated_df.csv',
                        type=str,
                        help="Path to the feature data CSV file.")
    parser.add_argument("--input", "-i",
                        type=str,
                        required=True,
                        choices=["features", "time-series"],
                        help="Type of input data ('features' or 'time-series').")
    parser.add_argument("--mapping",
                        type=bool,
                        default=False,
                        help="Labels to use: `False` for anomaly kinds, `True` for algorithm families.")
    parser.add_argument("--output", "-o",
                        type=str,
                        required=True,
                        help="Output file for saving the classification report.")

    # Hyperparameters
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001)
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--num_epochs",
                        type=int,
                        default=200)

    args = parser.parse_args()

    main(args.features, args.labels, args.input, args.output, (args.learning_rate, args.batch_size, args.num_epochs), args.mapping)
