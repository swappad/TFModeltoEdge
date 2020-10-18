import argparse
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from unet import unet

tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

parser = argparse.ArgumentParser(description='convert unet Model with h5 encoded weights to pb')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()



with tf.Session(graph=tf.Graph()) as sess:

    tf.keras.backend.set_learning_phase(0)
    model = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-3, lr_decay=5e-4)
    model.load_weights(args.weights)

    directory = './trained_models/'
    filename = '1210_unet.pb'
    output_graph = os.path.join(directory, filename)
    output_node_names = 'Identity'

    print([node.op.name for node in model.outputs])
    output__graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            [node.op.name for node in model.outputs]
            #output_node_names.split(",")
    )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output__graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output__graph_def.node))
    print("##### Model saved to " + os.path.splitext(args.weights)[0] + ".pb" + " #####")
    print(model.summary())
    
    



