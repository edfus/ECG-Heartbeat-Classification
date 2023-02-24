import numpy as np

from models import cnn
from dataset import training_set

from sklearn.model_selection import KFold, cross_val_score

BATCH_SIZE=50
EPOCHES=20

k_fold = KFold(n_splits=5)
fold_no = 0

# Per-fold score containers
acc_per_fold = []
loss_per_fold = []
separator_bar = '-' * 62
for train_indices, test_indices in k_fold.split(training_set.k_x_train):
  model = cnn.Model()

  model.compile(
      optimizer="Adam",
      loss="sparse_categorical_crossentropy",
      metrics=[
        "sparse_categorical_accuracy"
      ],
  )

  print(separator_bar)
  print(f'Training for fold {fold_no} ...')

  X_train, X_test = training_set.k_x_train[train_indices], training_set.k_x_train[test_indices]
  y_train, y_test = training_set.k_y_train[train_indices], training_set.k_y_train[test_indices]

  history = model.fit(
      X_train,
      y_train,
      epochs=EPOCHES,
      batch_size=BATCH_SIZE,
      verbose=0
  )

  scores = model.evaluate(
      X_test,
      y_test,
      verbose=0
  )

  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])
  fold_no += 1


# == Provide average scores ==
print(separator_bar)
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print(separator_bar)
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print(separator_bar)
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(separator_bar)