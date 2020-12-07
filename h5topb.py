import argparse
import os
import tensorflow as tf
from models.unet import unet
import keras as keras


parser = argparse.ArgumentParser(description='convert unet Model with h5 encoded weights to pb')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()

keras.backend.clear_session()
keras.backend.set_learning_phase(0)
with keras.backend.get_session() as sess:

    model = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-3, lr_decay=5e-4)
    model.load_weights(args.weights)


    # save frozen subset graph for acceleration
    output_graph = os.path.splitext(args.weights)[0] #+ ".pb"

    input_names=[out.op.name for out in model.inputs]
    print(input_names)
    output_names=[out.op.name for out in model.outputs]
    print('input  node is{}'.format(input_names))
    print('output node is{}'.format(output_names))

    saver = tf.train.Saver()
    graph_def = sess.graph.as_graph_def()
    # for node in graph_def.node:
    #    print(node)
    
    save_path = saver.save(sess, os.path.join(output_graph, "float_model.ckpt"))

    tf.train.write_graph(graph_def, output_graph, "infer_graph.pb", as_text=False)

    print("##### Model saved to " + os.path.splitext(args.weights)[0] + ".pb" + " #####")
    print(model.summary())
    
    



