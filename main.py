from models import cnn
from dataset import testing_set, training_set

from submit import submit

model = cnn.NeutralNetwork()

# Adam is a replacement optimization algorithm for 
# stochastic gradient descent for training deep 
# learning models. Adam combines the best properties 
# of the AdaGrad and RMSProp algorithms to provide 
# an optimization algorithm that can handle sparse 
# gradients on noisy problems.
model.compile(
    # Controlling overfitting, etc.
    optimizer="Adam",
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
BATCH_SIZE=50
EPOCHES=30

# https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer
# In my experience it usually not necessary to 
# do learning rate decay with Adam optimizer.
# The theory is that Adam already handles learning 
# rate optimization
history = model.fit(
    training_set.k_x_train,
    training_set.k_y_train,
    epochs=EPOCHES,
    batch_size=BATCH_SIZE,
    verbose=0
)

predictions = model.predict(testing_set.x_test)

submit(predictions)