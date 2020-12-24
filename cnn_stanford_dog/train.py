import tensorflow as tf

save_weight = "test"

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accurary = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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



if __name__ == "__main__":
    from tqdm import tqdm
    from foodnet import FoodNet

    model = FoodNet(4)

    x = torch.experimental.numpy.random.randn(1,3,224, 224)
    output = model(x)

    epochs = 10
    for epoch in range(epochs):
        with tf.device("CPU"):
            with tqdm(total = x_train.shape[0]) as pb:
                for data, target in train_ds:
                    train_step(data, target)

                for val_data, val_target in val_ds:
                    valid_step(val_data, val_target)

                for test_data, test_target in test_ds:
                    test_step(test_data, test_target)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, val_loss: {}, val_Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(
            epoch+1,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100

        ))

