import numpy as np
import os

from imblearn.over_sampling import SMOTE

from .helpers import import_data

__dirname = os.path.dirname(os.path.realpath(__file__))
train = import_data(os.path.join(__dirname, "data/train.csv"))

y_train = train["label"]
x_train = train.drop(["id", "label"], axis=1)

# upon inspection, we can say that this dataset 
# is characterized by having highly imbalanced classifications
# in which the number of examples in one class greatly outnumbers
# the examples in another.
# (e.g. label 0 (Normal) and label 1 (Fusion of ventricular and normal))

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

# 使用 SMOTE 对数据进行上采样以解决类别不平衡问题
smote = SMOTE(random_state=2021, n_jobs=-1)
k_x_train, k_y_train = smote.fit_resample(x_train, y_train)

# 将训练集转换为适应 CNN 输入的 shape
k_x_train = np.array(k_x_train).reshape(k_x_train.shape[0], k_x_train.shape[1], 1)
