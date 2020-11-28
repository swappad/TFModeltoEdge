import os
import sys
import argparse
import shutil
import tensorflow as tf
from tensorflow import keras
import cv2

from unet import unet
from dataset_parser.generator import data_generator

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

MODEL_WEIGHTS='../trained_models/unet_model_next.h5'
BATCH_SIZE=10

def model_fn():
    model = unet(input_shape=(256, 512, 3), num_classes=4,
                   lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=None)
    model.load_weights()

    estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model,
            checkpoint_format=saver)
    return estimator

def eval_input_fn():
    # for x, y in data_generator('dataset_parser/data.h5', BATCH_SIZE, 'val'):
    #     print(x.dtype, y.dtype)
    dataset = tf.data.Dataset.from_generator(
            lambda: data_generator('dataset_parser/data.h5', BATCH_SIZE, 'val'),
            (tf.float32, tf.float64))

    data = dataset.batch(1)
    iterator = data.make_one_shot_iterator()
    x, y = iterator.get_next()

    return tf.estimator.inputs.numpy_input_fn(
            x={"x": tf.squeeze(x,0)},
            y=tf.squeeze(y,0),
            num_epochs=1,
            shuffle=False)




eval_input_fn()

