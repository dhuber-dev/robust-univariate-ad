import argparse
import os
import yaml
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def read_yaml(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_results(output_path, results):
    """Save performance results to a YAML file."""
    results_file = os.path.join(output_path, "performance_results.yaml")
    with open(results_file, 'w') as file:
        yaml.dump(results, file)
    print(f"Results saved to {results_file}")


def preprocess_data(features, labels, time_steps=1000):
    """
    Preprocess the data to fit the time series format.
    - Assumes `features` is a DataFrame where each row is a flattened time series.
    - Reshapes the features into (num_samples, time_steps, 1).
    """
    # Reshape features into (num_samples, time_steps, 1) for 1D time series
    features = features.values.reshape(-1, time_steps, 1)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return features, labels, label_encoder


def build_model(input_shape, num_classes):
    """
    Build a model to handle time series data.
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer for classification
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def preprocess_exploded_features(df):
    """
    Preprocess the exploded features DataFrame into a wide format suitable for training.
    - Assumes `df` has columns: `id`, `time_idx`, and `value`.
    """
    # Pivot the DataFrame
    wide_df = df.pivot(index='id', columns='time_idx', values='value')

    # Reset the column names (ensure they are numerical for model input)
    wide_df.columns = [f"Feature_{col}" for col in wide_df.columns]
    wide_df.reset_index(drop=True, inplace=True)  # Remove the `id` column

    return wide_df



def main(features, labels, output_path):
    # Preprocess the data
    X, y, label_encoder = preprocess_data(features, labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(label_encoder.classes_))

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes, zero_division=0)
    print("Accuracy:", accuracy)
    print(class_report)

    # Save the model
    model_path = os.path.join(output_path, "trained_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save evaluation metrics
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "accuracy_score": accuracy,
        "classification_report": class_report
    }
    save_results(output_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DL Model for time series data.")
    parser.add_argument("--loaded-df", type=str)
    parser.add_argument("--exploded-df", type=str)
    parser.add_argument("--output-path",  default='results_dl_model_time_series', type=str)

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_path, exist_ok=True)

    data = pd.read_csv(args.exploded_df, index_col=0)
    data_wide = preprocess_exploded_features(data)

    label_data = pd.read_csv(args.loaded_df, index_col=0)

    main(data_wide, label_data.anomaly, args.output_path)
