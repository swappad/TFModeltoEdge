import argparse
import cv2
import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from unet import unet
from tensorflow import keras
#from tensorflow.python.client import session


# tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

parser = argparse.ArgumentParser(description='convert unet Model with h5 encoded weights to pb')
parser.add_argument("--weights", help="path to h5 encoded weights")
args = parser.parse_args()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=False):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
#        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph, protected_nodes=None)
    return frozen_graph


keras.backend.clear_session()
with keras.backend.get_session() as sess:
    keras.backend.set_learning_phase(0)

    model = unet(input_shape=(256, 512, 3), num_classes=4, lr_init=1e-3, lr_decay=5e-4)
    model.load_weights(args.weights)

    directory = './trained_models/'
    filename = '1210_unet'

#    # save complete model for comparison
#    print(model.inputs)
#    model.save(directory+filename)

    # save frozen subset graph for acceleration
    output_graph = os.path.splitext(args.weights)[0] #+ ".pb"
    output_names= ['conv2d_8/Conv2D']

    input_names=[out.op.name for out in model.inputs]
    output_names=[out.op.name for out in model.outputs]
    print('input  node is{}'.format(input_names))
    print('output node is{}'.format(output_names))


#    frozen_graph = freeze_session(session=sess, output_names=output_names)
    saver = tf.train.Saver()
#    saver = tf.compat.v1.train.Saver()
    graph_def = sess.graph.as_graph_def()

    save_path = saver.save(sess, os.path.join(output_graph, "float_model.ckpt"))

    tf.train.write_graph(graph_def, output_graph, "infer_graph.pb", as_text=False)


    #    print([node.op.name for node in model.outputs])
    #    
        
     #   output__graph_def = tf.graph_util.convert_variables_to_constants(
     #           sess,
     #           tf.get_default_graph().as_graph_def(),
     #           [node.op.name for node in model.outputs]
     #           #output_node_names.split(",")
     #   )

     #   with tf.gfile.GFile(output_graph, "wb") as f:
     #       f.write(output__graph_def.SerializeToString())

        # freeze_graph(os.path.splitext(args.weights)[0], sess.graph, sess, [node.op.name for node in model.outputs])



    #    print("%d ops in the final graph." % len(output__graph_def.node))


    print("##### Model saved to " + os.path.splitext(args.weights)[0] + ".pb" + " #####")
    print(model.summary())
    
    



