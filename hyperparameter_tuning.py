import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tsfresh import select_features

from FeedForward import FFModel
from evaluation import main, read_yaml, shape_data_to_model, evaluate_model, markdown_report


def get_report(report_name, y_test, y_pred):
    print(classification_report(y_test, y_pred, zero_division=0))
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    report = markdown_report(class_report)

    # Save the markdown report to a .md file
    with open(report_name, "w") as f:
        f.write(report)


def get_confusion_matrix(classification_matrix_name, y_test, y_pred):
    # -- Confusion Matrix --
    confusion_matrix = pd.crosstab(pd.Series(y_test, name='test'), pd.Series(y_pred, name='predicted'))
    confusion_matrix.to_csv(classification_matrix_name)


def hyperparameter_tuning(features, labels, learning_rate, batch_size):
    # Labels
    data = pd.read_csv(features, index_col=0)
    labels = pd.read_csv(labels, index_col=0)
    labels = labels.loc[~labels.invalid_data, 'anomaly'].squeeze()
    labels = labels.map(read_yaml('best_performer_mapper.yaml'))
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Features
    features = data.dropna(axis=1)
    features = features.loc[:, features.nunique() > 1]  # Remove columns with just one value
    features = features.clip(lower=-1e6, upper=1e6)
    features = select_features(features, labels)
    print(len(features.columns), 'features used.')

    data_loaders = shape_data_to_model('ff', features, labels, batch_size, True)
    _, _, test_loader = data_loaders
    y_test = test_loader.dataset.tensors[1]  # test labels in string form

    model = FFModel(input_shape=(data_loaders[0].batch_size, features.shape[1]),
                    num_classes=label_encoder.classes_.size)

    hyperparameters = (learning_rate, batch_size, 200)
    y_pred, loss_evaluation = evaluate_model(model, data_loaders, hyperparameters)

    output_folder = Path('results/hyperparameter_tuning/')
    output_folder.mkdir(exist_ok=True)

    base = f'lr_{str(learning_rate).replace(".", "p")}_bs_{batch_size}'
    get_report(output_folder / Path(f'{base}.md'), y_test, y_pred)
    pd.DataFrame(loss_evaluation).to_csv(output_folder / Path(f'{base}_loss_evaluation.csv'))
    get_confusion_matrix(output_folder / Path(f'{base}_confusion_matrix.csv'), y_test, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a model for time series data.")
    parser.add_argument("--labels", "-l",
                        default='loaded_self_generated_df.csv',
                        type=str,
                        help="Path to the label CSV file.")
    parser.add_argument("--features", "-f",
                        default='features_new.csv',
                        type=str,
                        help="Path to the feature data CSV file.")
    parser.add_argument("--bs",
                        default=64,
                        type=int)
    parser.add_argument("--lr",
                        default=0.001,
                        type=float)
    parser.add_argument('--learning-rate', dest='loop_learning_rate', action='store_true')

    args = parser.parse_args()

    # Hyperparameter space
    hyperparameter_space = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "batch_size": [pow(2, x) for x in range(1, 11)],
    }

    if args.loop_learning_rate:
        for lr in hyperparameter_space['learning_rate']:
            hyperparameter_tuning(args.features, args.labels, lr, args.bs)
    else:
        for bs in hyperparameter_space['batch_size']:
            hyperparameter_tuning(args.features, args.labels, args.lr, bs)
