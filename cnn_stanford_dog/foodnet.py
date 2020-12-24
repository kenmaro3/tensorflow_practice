import tensorflow as tf
from tensorflow.keras import layers, Model

class FoodNet(Model):
    def __init__(self, output_size):
        super(FoodNet, self).__init__()
        self.relu = layers.ReLU()
        self.conv1 = layers.Conv2D(12, kernel_size=7, strides=4, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(12, kernel_size=3, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(46, kernel_size=3, strides=1, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.bn5 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.bn6 = layers.BatchNormalization()
        self.conv7 = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.bn7 = layers.BatchNormalization()
        self.conv8 = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.bn8 = layers.BatchNormalization()

        self.gap = layers.GlobalAveragePooling2D()

        self.fc1 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(output_size, activation="softmax")


    def call(self, x):

        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn1(self.conv1(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn2(self.conv2(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn3(self.conv3(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn4(self.conv4(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn5(self.conv5(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn6(self.conv6(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn7(self.conv7(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.relu(self.bn8(self.conv8(x)))
        #print(f'x.shape >>> {x.shape}')
        x = self.gap(x)
        #print(f'x.shape >>> {x.shape}')

        x = self.fc1(x)
        #print(f'x.shape >>> {x.shape}')
        x = self.dropout(x)
        #print(f'x.shape >>> {x.shape}')
        x = self.fc2(x)
        #print(f'x.shape >>> {x.shape}')

        return x


if __name__ == "__main__":
    model = FoodNet(2)
    x = tf.experimental.numpy.random.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
