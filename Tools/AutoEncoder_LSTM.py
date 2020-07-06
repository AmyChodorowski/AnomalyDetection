import pandas as pd
import numpy as np
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt

import plotly.graph_objects as go

rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted')


class AutoEncoder_LSTM:

    def __init__(self, p_data, num_lstm_layers=128, num_timesteps=30, num_features=1):

        # Further reading: https://machinelearningmastery.com/lstm-autoencoders/
        # Look closely at Prediction LSTM Autoencoder
        self.p_data = p_data
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

        self.trained = False
        self.tested = False

        self.train_history = None
        self.train_loss = None
        self.test_loss = None
        self.threshold = None

    def train_model(self, num_patience=3):
        """
        Training the model on the preprocessed training data
        :return: Updates self.model_trained with fitted tensor model
        """

        x_train, y_train = self.p_data.create_sequences(self.p_data.train, self.num_timesteps)

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

        # Calculate the train losses
        x_train_pred = self.model.predict(x_train)
        train_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        # Determine the threshold
        m, s = np.mean(train_loss), np.std(train_loss)
        self.threshold = [m-s*3, m+s*3]

        self.train_loss = pd.DataFrame(self.p_data.train[self.num_timesteps:])
        self.train_loss['loss'] = train_loss
        self.train_loss['threshold_lower'] = self.threshold[0]
        self.train_loss['threshold_upper'] = self.threshold[1]

        self.trained = True

    def test_model(self):

        x_test, _ = self.p_data.create_sequences(self.p_data.test, self.num_timesteps)

        # Calculate test losses
        x_test_pred = self.model.predict(x_test)
        test_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

        self.test_loss = pd.DataFrame(self.p_data.test[self.num_timesteps:])
        self.test_loss['loss'] = test_loss
        self.test_loss['threshold_lower'] = self.threshold[0]
        self.test_loss['threshold_upper'] = self.threshold[1]
        self.test_loss['anomaly'] = (self.test_loss.threshold_lower > self.test_loss.loss) |\
                                    (self.test_loss.loss > self.test_loss.threshold_upper)

        self.tested = True

    def plot_training_history(self):

        if self.trained:
            print("Below is the training loss vs the validation loss for each epoch of the fitting the model to the "
                  "training data.")
            print("Validation loss << Training loss - Overfitting")
            print("Validation loss <= Training loss - Underfitting")
            plt.plot(self.train_history.history['loss'], label='Training loss')
            plt.plot(self.train_history.history['val_loss'], label='Validation loss')
            plt.xlabel('Epoch')
            plt.xlabel('Loss')
            plt.legend()

        else:
            raise ValueError('Model must be trained first with preprocessed data')

    def plot_trained_threshold(self):

        if self.trained:
            print("Below is the distribution of the training losses |(predicted - actual)|")
            print("Thresholds have been created using \u03BC \u00B1 3\u03C3")
            print("")
            print('Lower threshold: {l}'.format(l=self.threshold[0]))
            print('Upper threshold: {u}'.format(u=self.threshold[1]))

            sns.distplot(self.train_loss.loss, bins=50, kde=True, axlabel="Training loss")

        else:
            raise ValueError('Model must be trained first with preprocessed data')

    def plot_trained_loss(self):

        if self.trained:
            print("Below is train losses |(predicted - actual)| in time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.train_loss.date, y=self.train_loss.loss,
                                     mode='lines', name='Train Loss'))
            fig.add_trace(go.Scatter(x=self.train_loss.date, y=self.train_loss.threshold_lower,
                                     mode='lines', name='Threshold (lower)'))
            fig.add_trace(go.Scatter(x=self.train_loss.date, y=self.train_loss.threshold_upper,
                                     mode='lines', name='Threshold (upper)'))
            fig.update_layout(showlegend=True, xaxis_title="Date", yaxis_title="Loss")
            fig.show()

        else:
            raise ValueError('Model must be trained first with preprocessed data')

    def plot_tested_loss(self):

        if self.tested:
            print("Below is test losses |(predicted - actual)| in time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.test_loss.date, y=self.test_loss.loss,
                                     mode='lines', name='Test Loss'))
            fig.add_trace(go.Scatter(x=self.test_loss.date, y=self.test_loss.threshold_lower,
                                     mode='lines', name='Threshold (lower)'))
            fig.add_trace(go.Scatter(x=self.test_loss.date, y=self.test_loss.threshold_upper,
                                     mode='lines', name='Threshold (upper)'))
            fig.update_layout(showlegend=True, xaxis_title="Date", yaxis_title="Loss")
            fig.show()

        else:
            raise ValueError('Model must be tested')

    def plot_anomalies(self):

        if self.tested:

            print("Below is data set split into train and test, with anomalies marked")
            anomalies = self.test_loss[self.test_loss.anomaly]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.p_data.raw.date, y=self.p_data.raw.value,
                                     mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=self.test_loss.date, y=self.p_data.scaler.inverse_transform(self.test_loss.value),
                                     mode='lines', name='Tested'))
            fig.add_trace(go.Scatter(x=anomalies.date, y=self.p_data.scaler.inverse_transform(anomalies.value),
                                     mode='markers', name='Anomalies'))
            fig.update_layout(showlegend=True, xaxis_title="Date", yaxis_title="Value")
            fig.show()

        else:
            raise ValueError('Model must be tested')

