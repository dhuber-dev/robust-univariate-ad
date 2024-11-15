import argparse
import os
import yaml
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
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


def main(features, labels, output_path):
    X = features.dropna(axis=1)
    X = X.clip(lower=-1e6, upper=1e6)

    y = labels

    # Encode anomaly kind
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for classification
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # multi-class
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
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
    parser = argparse.ArgumentParser(description="DL Model that takes pre-extracted features as input.")
    parser.add_argument("--loaded-df", type=str)
    parser.add_argument("--features", type=str)
    parser.add_argument("--output-path", default='results_dl_model_features', type=str)

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_path, exist_ok=True)

    data = pd.read_csv(args.loaded_df, index_col=0)
    full_feature_set = pd.read_csv(args.features, index_col=0)

    if len(full_feature_set) != len(data):
        raise ValueError
    else:
        main(full_feature_set, data.anomaly, args.output_path)
