import os
from .helpers import import_data

__dirname = os.path.dirname(os.path.realpath(__file__))
train = import_data(os.path.join(__dirname, "data/train.csv"))

if __name__ == "__main__":
  from matplotlib import pyplot as plt

  plt.hist(train['label'])
  plt.title('A summary with the 4 classes of beat subtypes in the training data')
  plt.ylabel('Number of Beats')
  plt.xlabel('Heartbeat Subtype')
  plt.xticks(range(4))
  plt.savefig("matplotlib.png")
  print("Figure saved to matplotlib.png")
  exit()


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


import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy="not majority", n_jobs=-1) # using all processors
# https://arxiv.org/pdf/1106.1813.pdf
# under = RandomUnderSampler(sampling_strategy="auto")
steps = [('o', over)]
pipeline = Pipeline(steps=steps)
k_x_train, k_y_train = pipeline.fit_resample(x_train, y_train)

# Make it CNN compatible
k_x_train = np.array(k_x_train).reshape(k_x_train.shape[0], k_x_train.shape[1], 1)
