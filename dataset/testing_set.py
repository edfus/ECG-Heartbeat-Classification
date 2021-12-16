import numpy as np
import os

from .helpers import import_data

__dirname = os.path.dirname(os.path.realpath(__file__))
test = import_data(os.path.join(__dirname, "data/testA.csv"))

x_test = test.drop(["id"], axis=1)

# Prepare the input for CNN prediction
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)
