import numpy as np
import keras
import tensorflow as tf
#from tf.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt

from foodnet import FoodNet
from data import dispense_dataset


def scheduler(epoch):
  if epoch < 20:
    return 0.01
  elif epoch >= 20 and epoch < 40:
    return 0.001
  elif epoch >= 40 and epoch < 60:
    return 0.0001
  else:
    return 0.0001


def visualize(x,y, idx):
  #figure = plt.figure(figsize=(6,4))
  
  #for i in range(len(idx)):
  #  axis = figure.add_subplot(2,2,i+1)
  #  axis.imshow(x[idx[i]])
  #  axis.title = str(y[idx[i]])


  
  plt.imshow(x[idx])
  plt.title(str(y[idx]))
  plt.show()


if __name__ == "__main__":
  size = None
  model_path = "test.h5"

  train_x, train_y, test_x, test_y = dispense_dataset(return_np=True, size=size)
  
  #idx = 4
  #visualize(train_x, train_y, idx)


  model = FoodNet(4)

  model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']  
  )

  
  callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

  history = model.fit(train_x, train_y,
                      batch_size=32,
                      epochs=40,
                      verbose=1,
                      callbacks=[callbacks],
                      validation_data=(test_x, test_y)
                     )

  model.save(model_path)
 

