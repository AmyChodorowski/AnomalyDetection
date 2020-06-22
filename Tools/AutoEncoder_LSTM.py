import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 14, 8


class AutoEncoder_LSTM:

    def __init__(self, num_lstm_layers=128, num_timesteps=30, num_features=1):

        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.model = Sequential([
            # Encoder
            LSTM(num_lstm_layers, input_shape=(num_timesteps, num_features)),
            Dropout(0.2),

            RepeatVector(num_timesteps),

            # Decoder
            LSTM(num_lstm_layers, return_sequences=True),
            Dropout(0.2),

            # Return output in the right shape
            TimeDistributed(Dense(num_features))
        ])

        # Compile
        # mae = Mean Absolute Square
        # adam = Stochastic Gradient Descent method
        self.model.compile(loss='mae', optimizer='adam')

        self.train_history = None
        self.train_loss = None

    def train_model(self, p_data, num_patience=3):
        """
        Training the model on the preprocessed training data
        :param p_data: Class Preprocessor, preprocessed data
        :return: Updates self.model_trained with fitted tensor model
        """

        x_train, y_train = p_data.create_sequences(p_data.train, self.num_timesteps)

        # Early stopping
        es = EarlyStopping(monitor='val_loss', patience=num_patience, mode='min')
        history = self.model.fit(
            x_train, y_train,
            epochs=100,
            validation_split=0.1,   # 90% to fit, 10% to validate
            callbacks=[es],
            shuffle=False           # Time series data
        )

        self.train_history = history

        # Loss
        x_train_pred = self.model.predict(x_train)
        train_loss = pd.Data

    def plot_trained_model(self):

        if self.model_trained:
            plt.plot(self.train_history.history['loss'], label='Training loss')
            plt.plot(self.train_history.history['val_loss'], label='Validation loss')
            plt.legend()

        else:
            raise ValueError('Train model with data')
