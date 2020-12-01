import argparse
import cv2
import os
import tensorflow as tf
from models.unet import *
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from dataset_parser.generator import data_generator

import tensorflow_model_optimization as tfmot
from callback import TrainCheck, Strip

#tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

EVAL_BATCH = 1
lr_init=1e-4
lr_decay=5e-4

parser = argparse.ArgumentParser(description='convert unet Model with h5 encoded weights to pb')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()


keras.backend.clear_session()
keras.backend.set_learning_phase(0)

## load model and model weights 
model = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-4, lr_decay=5e-4)
model.load_weights(args.weights)

## evaluate baseline accuracy
_, baseline_model_accuracy = model.evaluate_generator(
        data_generator('dataset_parser/data.h5', EVAL_BATCH, 'val'),
        steps=10,
        workers=1
        )

print("Baseline accuracy: ", baseline_model_accuracy)

## pruning
PRUNE_BATCH=4
EPOCH = 2
VALIDATION_SPLIT = 0.1

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)

# model_for_pruning.compile(optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=[dice_coef])

model_for_pruning.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
          loss='categorical_crossentropy',
          metrics=[dice_coef])

print(model_for_pruning.summary())

# Define pruning params
pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.1,
            final_sparsity=0.8, begin_step=0, end_step=1500 // PRUNE_BATCH),
        'block_size': (1,1),
        'block_pooling_type': 'MAX'
        }

callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./log', update_freq='epoch'),
        Strip('unet_next', './pruned_weigths'),
        ]

model_for_pruning.fit_generator(
        data_generator('dataset_parser/data.h5', PRUNE_BATCH, 'train'),
        steps_per_epoch=2500 // PRUNE_BATCH,
        validation_data=data_generator('dataset_parser/data.h5', EVAL_BATCH, 'val'),
        validation_steps=100 // EVAL_BATCH,
        callbacks=callbacks,
        epochs=100,
        verbose=1)

_, model_for_pruning_accuracy = model_for_pruning.evaluate_generator(
        data_generator('dataset_parser/data.h5', EVAL_BATCH, 'val'),
        steps=1,
        workers=2
        )


#print("Baseline accuracy: ", baseline_model_accuracy)
print("Pruned accuracy: ", model_for_pruning_accuracy)

   



