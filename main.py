import tensorflow as tf

from models import cnn
from dataset import testing_set, training_set

from submit import submit

model = cnn.NeutralNetwork()

# If your Y's are one-hot encoded, use categorical_crossentropy. Examples (for a 3-class classification): [1,0,0] , [0,1,0], [0,0,1]
# But if your Y's are integers, use sparse_categorical_crossentropy. Examples for above 3-class classification problem: [1] , [2], [3]
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(
    training_set.k_x_train, training_set.k_y_train,
    epochs=10, batch_size=64, verbose=1, callbacks=[callback]
)

predictions = model.predict(testing_set.x_test)

submit(predictions)