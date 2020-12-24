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



class CBR(Model):
    def __init__(self, filters, kernel_size, strides):
        super().__init__()

        self.layers_ = [
            layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                          padding="same", use_bias=True, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU()
            ]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class VGG16(Model):
    def __init__(self, ouput_size=1000):
        super().__init__()
        self.layers_ = [
            CBR(64 ,3, 1),
            CBR(64, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(128, 3, 1),
            CBR(128, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            layers.Flatten(),
            layers.Dense(4096),
            layers.Dense(4096),
            layers.Dense(output_size, activation="softmax")
        ]

        def call(self, inputs):
            for layer in self.layers_:
                inputs = layer(inputs)
            return inputs


class BottleneckResidual(Model):
    def __init__(self, inc, outc, strides=1):
        super().__init__()

        self.projection = inc != outc
        inter_ch = outc//4
        params = {
            "padding":"same",
            "kernel_initializer":"he_normal",
            "use_bias": True,
        }

        self.common_layers = [
            layers.Conv2D(inc, kernel_size=1, strides=strides, **params),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(inter_ch, kernel_size=3, stride=1, **params),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(outc, kernel_size=1, strides=1, **params),
            layers.BatchNormalization(),
        ]

        if self.projection:
            self.projection_layers = [
                layers.Conv2D(out_ch, kernel_size=1, strides=strides, **params),
                layers.BatchNormalization(),
            ]

        self.concat_layers = [layers.Add(), layers.ReLU()]


    def call(self, inputs):
        h1 = inputs
        h2 = inputs

        for layer in self.common_layers:
            h1 = layer(h1)

        if self.projection:
            for layer in self.projection:
                h2 = layer(h2)

        outputs = [h1, h2]
        for layer in self.convat_layers:
            outputs = layer(outputs)

        return outputs


class ResNet50(Model):
    '''
    resnet50

    conv*1
    resblock(conv*3) *3
    resblock(conv*3) *4
    resblock(conv*3) *6
    resblock(conv*3) *3
    dense * 1
    so conv*49, dense*1
    '''

    def __init__(self, output_size=1000):
        super().__init__()

        self.layers = [
            layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            BottleneckResidual(64, 256),
            BottleneckResidual(256, 256),
            BottleneckResidual(256, 256),
            BottleneckResidual(256, 512, 2),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 512),
            BottleneckResidual(512, 1024, 2),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 1024),
            BottleneckResidual(1024, 2048, 2),
            BottleneckResidual(2048, 2048),
            BottleneckResidual(2048, 2048),
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                output_size, activation="softmax", kernel_initializer="he_normal"
            ),

        ]

    def call(self, inputs):
        for layer in self.layers_:
            inputs = layer(inputs)
        return inputs



class ResNet101(ResNet50):
    '''
    ResNet101

    conv*1
    resblock(conv*3) *3
    resblock(conv*3) * 4
    resblock(conv*3) * 23
    resblock(conv*3) *3
    dense * 1
    '''

    def __init__(self, output_size=1000):
        super().__init__()

        self.layers_ = [
            layers.Conv2D(64, 7, 2, padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]

        out_ch = 256

        for i in range(3):
            in_ch = out_ch //4 if i ==0 else out_ch
            self.layers_.append(BottleneckResidual(in_ch, out_ch))

        out_ch = 512
        for i in range(4):
            in_ch = out_ch //4 if i==0 else out_ch
            strides=2 if i==0 else 1
            self.layers_.append(BottleneckResidual(in_ch, out_ch, strides))


        out_ch = 1024
        for i in range(23):
            in_ch = out_ch //4 if i==0 else out_ch
            strides = 2 if i ==0 else 1
            self.layers_.append(BottleneckResidual(in_ch, out_ch, strides))

        self.layers_ += [
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                output_size, activation="softmax", kernel_initializer="he_normal"
            ),
        ]



if __name__ == "__main__":
    model = FoodNet(2)
    x = tf.experimental.numpy.random.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)


