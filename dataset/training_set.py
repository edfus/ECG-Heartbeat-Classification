import numpy as np
import os

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

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

# Addressing imbalanced datasets:
# #Oversample the minority classes, undersample the majority classes
# https://imbalanced-learn.org/stable/over_sampling.html


over = SMOTE(sampling_strategy="not majority", n_jobs=-1) # using all processors
# https://arxiv.org/pdf/1106.1813.pdf
# under = RandomUnderSampler(sampling_strategy="auto")
steps = [('o', over)]
pipeline = Pipeline(steps=steps)
k_x_train, k_y_train = pipeline.fit_resample(x_train, y_train)

# Reshape to CNN compatible input
k_x_train = np.array(k_x_train).reshape(k_x_train.shape[0], k_x_train.shape[1], 1)
