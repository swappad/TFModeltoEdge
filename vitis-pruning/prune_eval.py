import os
import sys
import argparse
import shutil
import tensorflow as tf
from tensorflow import keras

from models.unet import unet
from dataset_parser.generator import data_generator

#tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

MODEL_WEIGHTS='../trained_models/unet_model_next.h5'
BATCH_SIZE=1

def model_fn():
    model = unet(input_shape=(256, 512, 3), num_classes=4,
                   lr_init=1e-4, lr_decay=5e-4, vgg_weight_path=None)
    model.load_weights(MODEL_WEIGHTS)

    estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model,
            checkpoint_format='checkpoint')
    return estimator

def eval_input_fn():
    dataset = tf.data.Dataset.from_generator(
            lambda: data_generator('dataset_parser/data.h5', BATCH_SIZE, 'val'),
            (tf.float32, tf.float64),
            ((BATCH_SIZE, 256, 512, 3), (BATCH_SIZE, 256, 512, 4)))

    dataset = dataset.map(lambda features, labels: ({'input_1': tf.squeeze(features,0)}, tf.squeeze(labels,0)))
    data = dataset.batch(2).repeat()
#  
    return data

#    return tf.compat.v1.estimator.inputs.numpy_input_fn(
#            x={"x": x},
#            y=tf.squeeze(y,0),
#            num_epochs=1,
#            shuffle=False)



