from tensorflow import keras
#from keras import backend
from keras.models import load_model
from unet import unet
import argparse
import os

parser = argparse.ArgumentParser(description='convert unet Model with h5 encoded weights to pb')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()

model = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-3, lr_decay=5e-4)
model.load_weights(args.weights)

model.save(os.path.splitext(args.weights)[0] + ".pb")

print("##### Model saved to " + os.path.splitext(args.weights)[0] + ".pb" + " #####")
print(model.summary())
