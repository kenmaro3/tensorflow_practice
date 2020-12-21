import os
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dropout, Flatten, Dense

class UNet(Model):
    def __init__(self, config):
        super().__init__()

        self.enc = Encoder(config)
        self.dec = Decoder(config)

        #optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        #loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)


    def call(self, x):
        z1, z2, z3, z4_dropout, z5_dropout = self.enc(x)
        y = self.dec(z1, z2, z3, z4_dropout, z5_dropout)
        return y

    @tf.function
    def train_step(self, x, t):
        with tf.GradientTape() as tape:
            y = self.call(x)
            v_loss = self.loss_object(t, y)
            self.valid_loss(v_loss)
            return y

    @tf.function
    def valid_step(self, x, t):
        y = self.call(x)
        v_loss = self.loss_object(t, y)
        self.valid_loss(v_loss)
        return y


class Encoder(Model):
    def __init__(self, config):
        super().__init__()

        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3,3), name="block1_conv1", activation="relu", padding="same")
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3,3), name="block1_conv2", padding="same")
        self.block1_bn = tf.keras.layers.BatchNormalization()
        self.block1_act = tf.keras.layers.ReLU()
        self.block1_pool = tf.keras.layers.MaxPooling2D((2,2), strides=None, name="block1_pool")

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3,3), name="block2_conv1", activation="relu", padding="same")
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3,3), name="block2_conv2", padding="same")
        self.block2_bn = tf.keras.layers.BatchNormalization()
        self.block2_act = tf.keras.layers.ReLU()
        self.block2_pool = tf.keras.layers.MaxPooling2D((2,2), strides=None, name="block2_pool")

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3,3), name="block3_conv1", activation="relu", padding="same")
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3,3), name="block3_conv2", padding="same")
        self.block3_bn = tf.keras.BatchNormalization()
        self.block3_act = tf.keras.layers.ReLU()
        self.block3_pool = tf.keras.layers.MaxPooling2D((2,2), strides=None, name="block3_pool")

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3,3), name="block4_conv1", activation="relu", padding="same")
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3,3), name="block4_conv2", padding="same")
        self.block4_bn = tf.keras.layers.BatchNormalization()
        self.block4_act = tf.keras.layers.ReLU()
        self.block4_dropout = tf.keras.layers.Dropout(0.5)
        self.block4_pool = tf.keras.layers.MaxPooling2D((2,2), strides=None, name="block4_pool")

        self.block5_conv1 = tf.keras.layers.Conv2D(1024, (3,3), name="block5_conv1", activation="relu", padding="same")
        self.block5_conv2 = tf.keras.layers.Conv2D(1024, (3,3), name="block5_conv2", padding="same")
        self.block5_bn = tf.keras.layers.BatchNormalization()
        self.block5_act = tf.keras.layers.ReLU()
        self.block5_dropout = tf.keras.layers.Dropout(0.5)


    def call(self, x):
        z1 = self.block1_conv1(x)
        z1 = self.block1_conv2(z1)
        z1 = self.block1_bn(z1)
        z1 = self.block1_act(z1)
        z1_pool = self.block1_pool(z1)

        z2 = self.block2_conv1(z1_pool)
        z2 = self.block2_conv2(z2)
        z2 = self.block2_bn(z2)
        z2 = self.block2_act(z2)
        z2_pool = self.block2_pool(z2)

        z3 = self.block3_conv1(z2_pool)
        z3 = self.block3_conv2(z3)
        z3 = self.block3_bn(z3)
        z3 = self.block3_act(z3)
        z3_pool = self.block3_pool(z3)

        z4 = self.block4_conv1(z3_pool)
        z4 = self.block4_conv2(z4)
        z4 = self.block4_bn(z4)
        z4 = self.block4_act(z4)
        z4_dropout = self.block4_dropout(z4)
        z4_pool = self.block4_pool(z4_dropout)

        z5 = self.block5_conv1(z4_pool)
        z5 = self.block5_conv2(z5)
        z5 = self.block5_bn(z5)
        z5 = self.block5_act(z5)
        z5_dropout = self.block5_dropout(z5)

        return z1, z2, z3, z4_dropout, z5_dropout



class Decoder(Model):
    def __init__(self, config):
        super().__init__()

        self.block6_up = tf.keras.layers.UpSampling2D(size=(2,2))
        self.block6_conv1 = tf.keras.layers.Conv2D(512, (2,2), name="block6_conv1", activation="relu", padding="same")
        self.block6_conv2 = tf.keras.layers.Conv2D(512, (3,3), name="block6_conv2", activation="relu", padding="same")
        self.block6_conv3 = tf.keras.layers.Conv2D(512, (3,3), name="block6_conv3", padding="same")
        self.block6_bn = tf.keras.layers.BatchNormalization()
        self.block6_act = tf.keras.layers.ReLU()

        self.block7_up = tf.keras.layers.UpSampling2D(size=(2,2))
        self.block7_conv1 = tf.keras.layers.Conv2D(256, (2,2), name="block7_conv1", activation="relu", padding="same")
        self.block7_conv2 = tf.keras.layers.Conv2D(256, (3,3), name="block7_conv2", activation="relu", padding="same")
        self.block7_conv3 = tf.keras.layers.Conv2D(256, (3,3), name="block7_conv3", padding="same")
        self.block7_bn = tf.keras.layers.BatchNormalization()
        self.block7_act = tf.keras.layers.ReLU()

        self.block8_up = tf.keras.layers.UpSampling2D(size=(2,2))
        self.block8_conv1 = tf.keras.layers.Conv2D(128, (2,2), name="block8_conv1", activation="relu", padding="same")
        self.block8_conv2 = tf.keras.layers.Conv2D(128, (3,3), name="block8_conv2", activation="relu", padding="same")
        self.block8_conv3 = tf.keras.layers.Conv2D(128 ,(3,3), name="block8_conv3", padding="same")
        self.block8_bn = tf.keras.layers.BatchNormalization()
        self.block8_act = tf.keras.layers.ReLU()


        self.block9_up = tf.keras.layers.UpSampling2D(size=(2,2))
        self.block9_conv1 = tf.keras.layers.Conv2D(128, (2,2), name="block9_conv1", activation="relu", padding="same")
        self.block9_conv2 = tf.keras.layers.Conv2D(128, (3,3), name="block9_conv2", activation="relu", padding="same")
        self.block9_conv3 = tf.keras.layers.Conv2D(128, (3,3), name="block9_conv3", padding="same")
        self.block9_bn = tf.keras.layers.BatchNormalization()
        self.block9_act = tf.keras.layers.ReLU()
        self.output_conv = tf.keras.layers.Conv2D(config.model.num_class, (1,1), name="otuput", activation="softmax")


    def call(self, z1, z3,z4_dropout, z5_dropout):
        z6_up = self.block6_up(z5_dropout)
        z6 = self.block6_conv1(z6_up)
        z6 = tf.keras.layers.concatenate([z4_dropout, z6], axis=3)
        z6 = self.block6_conv2(z6)
        z6 = self.block6_conv3(z6)
        z6 = self.block6_bn(b6)
        z6 = self.block6_act(z6)

        z7_up = self.block7_up(z6)
        z7 = self.block7_conv1(z7_up)
        z7 = tf.keras.concatenate([z3, z7], axis=3)
        z7 = self.block7_conv2(z7)
        z7 = self.block7_conv3(z7)
        z7 = self.block7_bn(z7)
        z7 = self.block7_act(z7)

        z8_up = self.block8_up(z7)
        z8 = self.block8_conv1(z8_up)
        z8 = tf.keras.layers.concatenate([z2, z8], axis=3)
        z8 = self.block8_conv2(z8)
        z8 = self.block8_conv3(z8)
        z8 = self.block8_bn(z8)
        z8 = self.block8_act(z8)

        z9_up = self.block9_up(z8)
        z9 = self.block9_conv1(z9_up)
        z9 = tf.keras.layers.concatenate([z1, z9], axis=3)
        z9 = self.block9_conv2(z9)
        z9 = self.block9_conv3(z9)
        z9 = self.block9_bn(z9)
        z9 = self.block9_act(z9)

        y = self.output_conv(z9)

        return y



class Trainer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_metric_loss = tf.keras.metrics.Mean()
        self.train_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.validation_metric_loss = tf.keras.metrics.Mean()
        self.validation_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            #y_pred = self.model(x_batch, training=True)
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_batch,  y_pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_metric_loss(loss)
        self.train_metric_accuracy(y_batch, y_pred)

    @tf.function
    def eval_step(self, x_batch, y_batch):
        #y_pred = self.model(x_batch, training=False)
        y_pred = self.model(x_batch)
        loss =self.loss_fn(y_pred, y_batch)
        self.validation_metric_loss(lossZ)
        self.validation_metric_accuracy(y_batchl, y_pred)


    def train(self, epochs, train_dataset, validation_dataset, verbose=True):
        for epoch in range(epochs):
            print("***"*10)
            print(f'epoch >>> {epoch}')
            for i, (x_batch, y_batch) in enumerate(train_dataset):
                self.train_step(x_batch, y_batch)

            train_loss =self.train_metric_loss.result().numpy()
            train_accuracy = self.train_metric_accuracyl.result().numpy()

            self.train_metric_loss.reset_states()
            self.train_metric_accuracy.reset_states()

            for x_batch, y_batch in validation_dataset:
                self.eval_step(x_batch, y_batch)

            validation_loss = self.validation_metric_loss.result().numpy()
            validation_accuracy = self.validation_metric_accuracy.result().numpy()

            self.validation_metric_loss.reset_states()
            self.validation_metric_accuracy.reset_states()


            if verbose:
                train_log = "epoch={}: train loss={:.3f}, val_loss={:.3f}, train_acc={:.3f}, val_acc={:.3f}".format(
                        epoch+1,
                        train_loss,
                        validation_loss,
                        train_accuracy,
                        validation_accuracy,
                        )
                print(train_log)

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(validation_loss)
            self.history["accurary"].append(train_accuracy)
            self.history["val_accuracy"].appened(validation_accuracy)


class MyCNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")
        self.pool3 = tf.keras.layers.MaxPooling2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    num_epochs = 10
    batch_size=128
    shuffle_buffer_size = 10000
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28, 28, 1)/255.0
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    model = MyCNN()
    optimizer = tf.keras.optimizers.Adam()
    trainer = Trainer(model, optimizer)

    t1 = time.time()
    trainer.train(num_epochs, train_dataset, test_dataset, verbose=True)
    t2 = time.time()
    print("time >>> {}".format(t2-t1))




