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

        # Dropout works by probabilistically removing, 
        # or “dropping out,” inputs to a layer, which 
        # may be input variables in the data sample or 
        # activations from a previous layer. It has the 
        # effect of simulating a large number of networks 
        # with a very different network structure and, 
        # in turn, making nodes in the network generally 
        # more robust to the inputs.
        self.dropout = Dropout(.5)
        self.flatten = Flatten()

        self.fc1 = Dense(units=256, activation="relu")
        self.fc21 = Dense(units=16, activation="relu")
        self.fc22 = Dense(units=256, activation="sigmoid")
        self.fc3 = Dense(units=4, activation="softmax")

    def call(self, inputs, training=None):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.conv4(inputs)
        inputs = self.max_pool1(inputs)

        inputs = self.conv5(inputs)
        inputs = self.conv6(inputs)
        inputs = self.max_pool2(inputs)

        # Some layers, in particular the BatchNormalization
        # layer and the Dropout layer, have different 
        # behaviors during training and inference. For such 
        # layers, it is standard practice to expose a training 
        # (boolean) argument in the call() method.
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models
        if training:
          inputs = self.dropout(inputs)
        inputs = self.flatten(inputs)

        x1 = self.fc1(inputs)
        x2 = self.fc22(self.fc21(inputs))
        inputs = self.fc3(x1 + x2)

        return inputs
