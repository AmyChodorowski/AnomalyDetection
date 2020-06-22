import pandas as pd
import numpy as np
import copy as cp

from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self, data, training_size=0.8):

        self.raw = cp.deepcopy(data)
        self.check_data()

        # Split training and testing
        train_index = int(len(self.raw) * training_size)
        self.train = self.raw.iloc[:train_index]
        self.test = self.raw.iloc[train_index:]

    def check_data(self):

        if type(self.raw) is not pd.DataFrame:
            raise ValueError('Dataset should be pandas DataFrame')
        elif self.raw.empty:
            raise ValueError('Dataset is empty')
        elif 'date' not in self.raw.columns:
            raise ValueError('Dataset does not contain date field')
        elif 'value' not in self.raw.columns:
            raise ValueError('Dataset does not contain value field')

    def standardise(self):
        # Removing the mean and scaling to unit variance, based from training
        scaler = StandardScaler()
        scaler = scaler.fit(self.train[['value']])

        # Standardise train and test
        self.train['value'] = scaler.transform(self.train[['value']])
        self.test['value'] = scaler.transform(self.test[['value']])

    @staticmethod
    def create_sequences(x, timesteps):
        """

        :param x: panda.DataFrame (N length) of chronological data values
        :param timesteps: int of how far back in time the xs values look back
        :return: xs = numpy.array (N-timesteps length) of numpy.array (timesteps length) of x, moving forward each time
                 ys = numpy.array (N-timesteps length) with the corresponding value for each respective xs array
        """
        xs, ys = [], []
        x_df = x[['value']]
        x_s = x.value
        for i in range(len(x) - timesteps):
            xs.append(x_df.iloc[i: i + timesteps].values)
            ys.append(x_s.iloc[i + timesteps])
        return np.array(xs), np.array(ys)

