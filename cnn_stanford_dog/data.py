import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

import os
from os.path import join as osp
import itertools
import time

import numpy as np

from PIL import Image

def pp_image(x_path):
  size = 128

  test1 = Image.open(x_path)

  test1 = test1.resize((size, size))
  test_np = np.array(test1)
  test_np = test_np/255.0
  return test_np


def label_to_onehot(y, class_size):
  test1 = np.zeros((len(y), class_size))
  for i in range(len(test1)):
    test1[i][y[i]] = 1.0

  return test1



def dispense_dataset(return_np=False, size=None):
  train_path = "./data_for_sikibetu"


  target_list = os.listdir(train_path)

  tmp1 = []
  for i, target in enumerate(target_list):
      file_list = os.listdir(osp(train_path, target))
      file_list2 = []

      if size is None:
        loop_size = len(file_list)
      else:
        loop_size = size
      for j in range(loop_size):
      #for j in range(10):
          file_list2.append(osp(train_path, target, file_list[j]))
      #file_list2 = [osp(train_path, target, file_) for file_ in file_list]

      tmp1.append(file_list2)

  tmp2 = []

  for i in range(len(tmp1)):
      tmp3 = []
      for j in range(len(tmp1[i])):
          tmp3.append(i)
      tmp2.append(tmp3)

  ds = list(itertools.chain.from_iterable(tmp1))
  ls = list(itertools.chain.from_iterable(tmp2))

  train_x, test_x, train_y, test_y = train_test_split(ds, ls, test_size=0.2)

  print(len(train_x))


  t1 = time.time()
  train_x_np = list(map(pp_image, train_x))
  train_x_np = np.array(train_x_np)

  train_y_np = np.array(train_y)

  test_x_np = list(map(pp_image, test_x))
  test_x_np = np.array(test_x_np)

  test_y_np = np.array(test_y)

  t2 = time.time()
  print(f'time >>> {t2-t1}')

  if not return_np:
    train_x_ds = tf.data.Dataset.from_tensor_slices(train_x_np)
    train_y_ds = tf.data.Dataset.from_tensor_slices(train_y_np)

    test_x_ds = tf.data.Dataset.from_tensor_slices(test_x_np)
    test_y_ds = tf.data.Dataset.from_tensor_slices(test_y_np)


    return train_x_ds, train_y_ds, test_x_ds, test_y_ds, len(train_x)

  else:
    train_y_np = label_to_onehot(train_y_np, len(tmp1))
    test_y_np = label_to_onehot(test_y_np, len(tmp1))
    return train_x_np, train_y_np, test_x_np, test_y_np



if __name__ == "__main__":
  batch_size = 32
  
  train_x_ds, train_y_ds, test_x_ds, test_y_ds = dispense_dataset()

  train_ds = tf.data.Dataset.zip((train_x_ds, train_y_ds))
  test_ds = tf.data.Dataset.zip((test_x_ds, test_y_ds))

  train_ds = train_ds.shuffle(200).batch(batch_size)
  test_ds = test_ds.batch(12)
