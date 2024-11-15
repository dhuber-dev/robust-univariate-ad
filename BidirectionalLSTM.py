from keras import Sequential, Input
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.src.layers import Bidirectional, LSTM


class BidirectionalLSTM:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the TimeSeriesModel with the given input shape and number of classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and compile an LSTM-based model for improved time-series performance.
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=self.input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')  # Output layer for classification
        ])

        # Adjust learning rate for better convergence
        optimizer = Adam(learning_rate=0.0005)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_model(self):
        """
        Return the compiled model.
        """
        return self.model