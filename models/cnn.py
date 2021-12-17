import tensorflow.keras as K
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Conv1D,
    MaxPool1D,
    Dropout
)

class Model(K.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(205, 1),
        )
        self.conv2 = Conv1D(
            filters=32,
            kernel_size=3,
            dilation_rate=2,
            padding="same",
            activation="relu",
        )
        self.conv3 = Conv1D(
            filters=64,
            kernel_size=3,
            dilation_rate=2,
            padding="same",
            activation="relu",
        )
        self.conv4 = Conv1D(
            filters=64,
            kernel_size=5,
            dilation_rate=2,
            padding="same",
            activation="relu",
        )
        self.max_pool1 = MaxPool1D(pool_size=3, strides=2, padding="same")

        self.conv5 = Conv1D(
            filters=128,
            kernel_size=5,
            dilation_rate=2,
            padding="same",
            activation="relu",
        )
        self.conv6 = Conv1D(
            filters=128,
            kernel_size=5,
            dilation_rate=2,
            padding="same",
            activation="relu",
        )
        self.max_pool2 = MaxPool1D(pool_size=3, strides=2, padding="same")

        # Controlling overfitting:
        # #2 dropout
        self.dropout = Dropout(.5)
        self.flatten = Flatten()

        self.fc1 = Dense(units=256, activation="relu")
        self.fc21 = Dense(units=16, activation="relu")
        self.fc22 = Dense(units=256, activation="sigmoid")
        self.fc3 = Dense(units=4, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool2(x)

        x = self.dropout(x)
        x = self.flatten(x)

        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        x = self.fc3(x1 + x2)

        return x
