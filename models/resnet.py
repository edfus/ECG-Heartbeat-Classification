# Taken from
# https://github.com/safwankdb/ResNet34-TF2/blob/master/model.py

from tensorflow import keras as K
from tensorflow.keras.layers import (
  Conv2D, BatchNormalization, ReLU,
  MaxPooling2D, GlobalAveragePooling2D, ReLU,
  Dense, Dropout
)
from tensorflow.keras import layers as Layers

# Batch size 32
# Epochs size 40

class ResBlock(K.Model):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__(name="ResBlock")
        self.flag = stride != 1
        self.conv1 = Conv2D(channels, 3, stride, padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, 3, padding="same")
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = K.Sequential()
    res_block.add(ResBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(ResBlock(filter_num, stride=1))

    return res_block


class ResNetTypeI(K.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()
        self.conv1 = Conv2D(64, 7, 2, padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.layer1 = make_basic_block_layer(filter_num=64, blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(
            filter_num=128, blocks=layer_params[1], stride=2
        )
        self.layer3 = make_basic_block_layer(
            filter_num=256, blocks=layer_params[2], stride=2
        )
        self.layer4 = make_basic_block_layer(
            filter_num=512, blocks=layer_params[3], stride=2
        )

        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation="relu")
        self.dp1 = Dropout(0.5)
        self.fc2 = Dense(512, activation="relu")
        self.dp2 = Dropout(0.5)
        self.fc3 = Dense(64)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.pool(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x


def ResNet18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def ResNet34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])
