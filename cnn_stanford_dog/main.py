from tqdm import tqdm

import tensorflow as tf
from data import dispense_dataset
from foodnet import FoodNet
#from train import train_step, val_step, test_step



if __name__ == "__main__":
    batch_size = 32

    train_x_ds, train_y_ds, test_x_ds, test_y_ds = dispense_dataset()

    model = FoodNet(4)

    train_x_ds, train_y_ds, test_x_ds, test_y_ds = dispense_dataset()

    train_ds = tf.data.Dataset.zip((train_x_ds, train_y_ds))
    test_ds = tf.data.Dataset.zip((test_x_ds, test_y_ds))

    train_ds = train_ds.shuffle(200).batch(batch_size)
    test_ds = test_ds.batch(12)


    save_weight = "test"

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accurary = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')




    epochs = 100

    for epoch in range(epochs):
        if epoch < 20:
            optimizer = tf.keras.optimizers.Adam(0.001)
        elif epoch >=20 and epoch < 40:
            optimizer = tf.keras.optimizers.Adam(0.0001)
        elif epoch >=40 and epoch < 60:
            optimizer = tf.keras.optimizers.Adam(0.00001)
        elif epoch >=60:
            optimizer = tf.keras.optimizers.Adam(0.000001)

        @tf.function
        def train_step(data, target):
            with tf.GradientTape() as tape:
                predictions = model(data, training=True)
                loss = loss_object(target, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(target, predictions)

        @tf.function
        def val_step(data, target):
            predictions = model(data)
            t_loss = loss_object(target, predictions)

            val_loss(t_loss)
            val_accuracy(target, predictions)

        @tf.function
        def test_step(data, target):
            predictions = model(data)
            t_loss = loss_object(target, predictions)

            test_loss(t_loss)
            test_accuracy(target, predictions)


        with tf.device("CPU"):
            with tqdm(total=len(train_x_ds)) as pb:
                for data, target in train_ds:
                    train_step(data, target)
                    pb.update(data.shape[0])
                for test_data, test_target in test_ds:
                    test_step(test_data, test_target)
            #template = 'Epoch {}, Loss: {}, Accuracy: {}, val_loss: {}, val_Acc: {}, Test Loss: {}, Test Acc: {}'
        template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Acc: {:.2f}'
        print(template.format(
            epoch+1,
            train_loss.result(),
            train_accuracy.result()*100,
            #val_loss.result(),
            #val_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100

        ))

