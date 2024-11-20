# main_script.py
import argparse
import os
from pathlib import Path

import torch.nn as nn
import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from CNN import TimeSeriesCNN
from FeedForward import FFModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


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


def shape_data_to_model(model, features, labels, batch_size, scaling):
    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
        X_test = X_test.to_numpy()

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    if model=='cnn':
        # Add 1D dimension for CNN
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, data_loaders, hyperparameters, early_stopping_patience=10):
    learning_rate, batch_size, num_epochs = hyperparameters
    train_loader, val_loader, test_loader = data_loaders

    # Set up model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_evolution = {
            'epoch': list(),
            'train_loss': list(),
            'val_loss': list(),
            'train_accuracy': list(),
            'val_accuracy': list()
        }

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop with validation and early stopping
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss and accuracy
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        # Compute average training loss and accuracy
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        # Compute average validation loss and accuracy
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        # Print epoch statistics
        print(
            f'Epoch {epoch + 1}/{num_epochs}, '
            f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
            f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}'
        )

        # Record loss evolution
        loss_evolution['epoch'].append(epoch + 1)
        loss_evolution['train_loss'].append(epoch_train_loss)
        loss_evolution['val_loss'].append(epoch_val_loss)
        loss_evolution['train_accuracy'].append(epoch_train_acc)
        loss_evolution['val_accuracy'].append(epoch_val_acc)

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Testing phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    num_elements = len(test_loader.dataset)
    y_pred = torch.zeros(num_elements, dtype=torch.long)

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)

            _, predicted = torch.max(outputs, 1)
            start = i * test_loader.batch_size
            end = start + X_batch.size(0)
            y_pred[start:end] = predicted.cpu()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    # Compute average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return y_pred, loss_evolution


def main(features, labels, input_type, output_path, hyperparameters, mapping, scaling, model):
    # Load data
    data = pd.read_csv(features, index_col=0)
    labels = pd.read_csv(labels, index_col=0)
    labels = labels.loc[~labels.invalid_data, 'anomaly'].squeeze()

    # Prepare
    ## Features
    if input_type == 'time-series':
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
        # Use algorithm family labels instead of anomaly kinds
        labels = labels.map(read_yaml('best_performer_mapper.yaml'))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    _, batch_size, _ = hyperparameters
    data_loaders = shape_data_to_model(model, features, labels, batch_size, scaling)

    # Select Model
    if model == 'ff':
        model = FFModel(input_shape=(data_loaders[0].batch_size, features.shape[1]),
                        num_classes=label_encoder.classes_.size)
    elif model == 'cnn':
        model = TimeSeriesCNN(num_classes=label_encoder.classes_.size)

    y_pred, loss_evaluation = evaluate_model(model, data_loaders, hyperparameters)

    loss_folder = Path('loss_evaluations/')
    (output_path.parent / loss_folder).mkdir(exist_ok=True)
    pd.DataFrame(loss_evaluation).to_csv(output_path.parent / loss_folder / Path(output_path.name.replace('.md', '_loss_evaluation.csv')))

    # Translate labels back to interpretable strings
    _, _, test_loader = data_loaders
    y_test = label_encoder.inverse_transform(test_loader.dataset.tensors[1])  # test labels in string form
    y_pred = label_encoder.inverse_transform(y_pred)  # predicted labels in string form

    def get_report(report_directory):
        print(classification_report(y_test, y_pred, zero_division=0))
        class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        report = markdown_report(class_report)

        learning_rate, batch_size, num_epochs = hyperparameters
        report += f'\nInput Type: {input_type}\nLearning Rate: {learning_rate}\nBatch Size: {batch_size}\nNumber of Epochs: {num_epochs}'
        report += f'\nMapping: {label_mapping}'

        # Save the markdown report to a .md file
        with open(report_directory, "w") as f:
            f.write(report)

        print(f'Report written to {report_directory.name}')

        # -- Confusion Matrix --
        confusion_matrix_folder = Path('confusion_matrices/')
        (report_directory.parent / confusion_matrix_folder).mkdir(exist_ok=True)
        confusion_matrix = pd.crosstab(pd.Series(y_test, name='test'), pd.Series(y_pred, name='predicted'))
        confusion_matrix.to_csv(report_directory.parent / confusion_matrix_folder / Path(report_directory.name.replace('.md', '_confusion_matrix.csv')))

    if mapping:
        get_report(output_path.with_name(output_path.name.replace('.md', '_algo_family.md')))
    else:
        get_report(output_path.with_name(output_path.name.replace('.md', '_anomaly_kind.md')))  # get report before translating to algo family
        # Translate anomaly kind labels to algorithm family labels
        mapper = read_yaml('best_performer_mapper.yaml')
        y_test = [mapper[x] for x in y_test]  # test labels in string form
        y_pred = [mapper[x] for x in y_pred]  # predicted labels in string form

        get_report(output_path.with_name(output_path.name.replace('.md', '_algo_family.md')))  # save another report with the final class labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a model for time series data.")
    parser.add_argument("--labels", "-l",
                        default='loaded_self_generated_df.csv',
                        type=str,
                        help="Path to the label CSV file.")
    parser.add_argument("--features", "-f",
                        type=str,
                        help="Path to the feature data CSV file.")

    parser.add_argument("--model", "-m",
                        type=str,
                        required=True,
                        choices=["ff", "cnn"],
                        help="Type of input data ('features' or 'time-series').")
    parser.add_argument("--input", "-i",
                        type=str,
                        required=True,
                        choices=["features", "time-series"],
                        help="Type of input data ('features' or 'time-series').")
    parser.add_argument('--pre-mapping', dest='mapping', action='store_true')
    parser.add_argument('--post-mapping', dest='mapping', action='store_false')
    parser.add_argument('--scaling', dest='scaling', action='store_true')
    parser.add_argument('--no-scaling', dest='scaling', action='store_false')

    parser.add_argument("--output", "-o",
                        type=str,
                        default='',
                        help="Output folder for saving the classification report and data (will be extended with test specifications).")

    # Hyperparameters
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001)
    parser.add_argument("--batch-size",
                        type=int,
                        default=64)
    parser.add_argument("--num-epochs",
                        type=int,
                        default=200)

    args = parser.parse_args()

    # Add text specifications to output file names
    mapping_label = 'pre-mapping' if args.mapping else 'post-mapping'
    scaling_label = 'scaling' if args.scaling else 'no-scaling'
    output_file_name = '_'.join([args.model, args.input, mapping_label, scaling_label]) + '.md'

    if not args.features:  # No feature source was provided
        args.features = 'exploded_self_generated_df.csv' if args.input == 'time-series' else 'features_new.csv'

    main(args.features, args.labels, args.input, Path(args.output + output_file_name), (args.learning_rate, args.batch_size, args.num_epochs), args.mapping, args.scaling, args.model)
