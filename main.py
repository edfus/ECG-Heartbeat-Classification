from tensorflow import keras

from models import cnn
from dataset import testing_set, training_set

from submit import submit

model = cnn.Model()

initial_learning_rate = 0.001 # Adam's
# Adam is a replacement optimization algorithm for 
# stochastic gradient descent for training deep 
# learning models. Adam combines the best properties 
# of the AdaGrad and RMSProp algorithms to provide 
# an optimization algorithm that can handle sparse 
# gradients on noisy problems.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    # The conventional way to store a matrix is to store 
    # all np of the values, even if most of those values 
    # are zeros.
    # The sparse matrix exploits the structure of the labels
    # to store fewer values.
    # So not bothering translating heartbeat subtype 
    # representations to one-hot vector here.
    loss="sparse_categorical_crossentropy",
    metrics=[
      "sparse_categorical_accuracy"
    ],
)

# Using a smaller batch size is like using some 
# regularization to avoid converging to sharp minimizers. 
# The gradients calculated with a small batch size are 
# much more noisy than gradients calculated with large 
# batch size, so it's easier for the model to escape 
# from sharp minimizers, and thus leads to a better 
# generalization.

# As with our case where the model tends to overfit, 
# the gradient noise added due to smaller mini-batches 
# act as a good regularizer. Decreasing the batch size 
# while decreasing the learning rate might lead to a 
# better result.
# edit: SIKE

# https://axon.cs.byu.edu/papers/Wilson.nn03.batch.pdf
# The general inefficiency of batch training for gradient descent learning
BATCH_SIZE=64
EPOCHS=30

from tensorflow import math
from tensorflow.keras.callbacks import LearningRateScheduler

# https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

decay_rate = 0.4

def lr_time_based_decay(epoch, lr):
    lr = initial_learning_rate * (1. / (1. + decay_rate * epoch))
    return lr

k = 0.1
def lr_exp_decay(epoch, lr):
    return initial_learning_rate * math.exp(-k * epoch)

decay_rate_exp = 0.9
def lr_exp_decay_2(epoch, lr):
    return initial_learning_rate * (decay_rate_exp ** epoch)

lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)

history = model.fit(
    training_set.k_x_train,
    training_set.k_y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[lr_scheduler]
)

print(history.history)

predictions = model.predict(testing_set.x_test)

print(
  "Prediction result is saved to '{}'".format(
    submit(predictions)
    )
)