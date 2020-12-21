from tqdm import tqdm

import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")


    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x



class EagerTrainer(object):
    def __init__(self):
        self.model = MyModel()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")



    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image)
            loss = self.loss_object(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)

    @tf.function
    def test_step(self, image, label):
        predictions = self.model(image)
        t_loss = self.loss_object(label, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(label, predictions)

    def train(self, epochs, training_data, test_data):
        template = "epochs {}, loss: {:.5f}, accuracy: {:.5f}, test loss: {:.5f}, test accuracy: {:.5f}, elapsed_time: {:.5f}"

        for epoch in range(epochs):
            start = time.time()
            for image, label in tqdm(training_data):
                self.train_step(image, label)
            elapsed_time = time.time() - start

            for test_image, test_label in test_data:
                self.test_step(test_image, test_label)

            print(template.format(epoch+1,
                                self.train_loss.result(),
                                self.train_accuracy.result()*100,
                                self.test_loss.result(),
                                self.test_accuracy.result()*100,
                                elapsed_time))


class KerasTrainer(object):
    def __init__(self):
        self.model = MyModel()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=["accuracy"])

    def train(self, epochs, training_data, test_data):
        self.model.fit(training_data, epochs=epochs, validation_data=test_data)


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label




if __name__ == "__main__":
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']

    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
    mnist_test = mnist_test.map(convert_types).batch(32)

    #trainer = KerasTrainer()
    trainer = EagerTrainer()


    trainer.train(epochs=5, training_data=mnist_train, test_data=mnist_test)





