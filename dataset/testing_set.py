import numpy as np
import os

from .helpers import import_data

__dirname = os.path.dirname(os.path.realpath(__file__))
test = import_data(os.path.join(__dirname, "data/testA.csv"))

x_test = test.drop(["id"], axis=1)

# 将测试集转换为适应 CNN 输入的 shape
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)
