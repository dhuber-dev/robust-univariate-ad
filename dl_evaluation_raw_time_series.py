# main_script.py
import argparse
import os
import yaml
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from FeedForward import FFModel
from BidirectionalLSTM import BidirectionalLSTM


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


def preprocess_data(features, labels, time_steps=1000):
    features = features.values.reshape(-1, time_steps, 1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return features, labels, label_encoder


def preprocess_exploded_features(df):
    wide_df = df.pivot(index='id', columns='time_idx', values='value')
    wide_df.columns = [f"Feature_{col}" for col in wide_df.columns]
    wide_df.reset_index(drop=True, inplace=True)
    return wide_df


def main(features, labels, output_path):
    X, y, label_encoder = preprocess_data(features, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile the model
    model_instance = BidirectionalLSTM(input_shape=(X_train.shape[1], X_train.shape[2]),
                                       num_classes=len(label_encoder.classes_))

    num_classes = len(labels.unique())
    xgb_model = XGBoostModel(num_classes)

    model = model_instance.get_model()

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Predictions and evaluation metrics
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes, zero_division=0)
    print("Accuracy:", accuracy)
    print(class_report)

    # Save model and results
    model_path = os.path.join(output_path, "trained_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "accuracy_score": accuracy,
        "classification_report": class_report
    }
    save_results(output_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DL Model for time series data.")
    parser.add_argument("-l", "--labels", type=str)
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-i", "--input", description="'features' or 'time-series'", type=str)
    parser.add_argument("-m", "--model", type=str, help="Name of the model class (e.g., FeedForward, LSTMTimeSeries, FeatureBased, XGBoost).")
    parser.add_argument("--output-path", default='results_dl_model_time_series', type=str)

    args = parser.parse_args()

    # Load the specified model class
    ModelClass = load_model_class(args.model)

    if args.model in ["FeedForward", "LSTMTimeSeries", "FeatureBased"]:
        if args.input_shape is None:
            raise ValueError("Input shape must be provided for neural network models.")
        model_instance = ModelClass(input_shape=tuple(args.input_shape), num_classes=args.num_classes)
    elif args.model == "XGBoost":
        model_instance = ModelClass(num_classes=args.num_classes)
    else:
        raise ValueError("Invalid model name.")

    print(f"Initialized model: {args.model} on {args.input}")

    os.makedirs(args.output_path, exist_ok=True)

    data = pd.read_csv(args.data, index_col=0)

    if args.input == 'time-series':
        data_wide = preprocess_exploded_features(data)

    label_data = pd.read_csv(args.loaded_df, index_col=0)

    main(data_wide, label_data.anomaly, args.output_path)
