#from __future__ import print_function

import cv2
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot

class TrainCheck(Callback):
    def __init__(self, output_path, model_name):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name

    def result_map_to_img(self, res_map):
        img = np.zeros((256, 512, 3), dtype=np.uint8)
        res_map = np.squeeze(res_map)

        argmax_idx = np.argmax(res_map, axis=2)

        # For np.where calculation.
        person = (argmax_idx == 1)
        car = (argmax_idx == 2)
        road = (argmax_idx == 3)

        img[:, :, 0] = np.where(person, 255, 0)
        img[:, :, 1] = np.where(car, 255, 0)
        img[:, :, 2] = np.where(road, 255, 0)

        return img

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        self.visualize('img/test.png')

    def visualize(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img = img / 127.5 - 1

        pred = self.model.predict(img)
        res_img = self.result_map_to_img(pred[0])

        cv2.imwrite(os.path.join(self.output_path, self.model_name + '_epoch_' + str(self.epoch) + '.png'), res_img)


class Strip(Callback):
    def __init__(self, model_name, strip_dir):
        self.epoch = 0
        self.model_name = model_name
        self.strip_dir = strip_dir

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
        tf.keras.models.save_model(model_for_export, os.path.join(self.strip_dir, self.model_name + '_epoch_' + str(self.epoch) + '.h5'), include_optimizer=False)
        print('Saved pruned Keras model to', self.strip_dir)
        epoch +=1


