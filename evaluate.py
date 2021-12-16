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
for train_indices, test_indices in k_fold.split(training_set.k_x_train):
  model = cnn.NeutralNetwork()

  '''
  from keras.callbacks import LearningRateScheduler
  def decay_schedule(epoch, lr):
      # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
      if (epoch % 5 == 0) and (epoch != 0):
          lr = lr * 0.1
      return lr

  lr_scheduler = LearningRateScheduler(decay_schedule)
  '''

  model.compile(
      optimizer="Adam",
      loss="sparse_categorical_crossentropy",
      metrics=[
        "sparse_categorical_accuracy"
      ],
  )

  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  X_train, X_test = training_set.k_x_train[train_indices], training_set.k_x_train[train_indices]
  y_train, y_test = training_set.k_y_train[test_indices], training_set.k_y_train[test_indices]

  history = model.fit(
      X_train,
      y_train,
      epochs=EPOCHES,
      batch_size=BATCH_SIZE,
      verbose=1
  )

  scores = model.evaluate(
      X_test,
      y_test,
      verbose=1
  )

  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')