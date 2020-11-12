import os
import sys
import argparse
import shutil
import glob
import tensorflow as tf
import numpy as np
import cv2

import tensorflow.contrib.decent_q
from tensorflow.python.platform import gfile

calib_img_path = "./dataset/leftImg8bit_trainvaltest/leftImg8bit/train"
calib_batch_size = 1

def result_map_to_img(res_map):
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


def graph_eval(input_graph_def, input_node, output_node):

    ## images for evaluation
    paths = []
    for (path, dirname, files) in sorted(os.walk(calib_img_path)):
      for filename in sorted(files):
          if filename.endswith(('.jpg', '.png')):
              paths.append(os.path.join(path, filename))

    images = []
    for index in range(0, calib_batch_size):
      #print("Path: " + paths[(iter * calib_batch_size + index) % len(paths)])
      img = cv2.imread(paths[(18 * calib_batch_size + index) % len(paths)], 1)
      try:
          img = cv2.resize(img,(512, 256))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = img / 127.5 - 1
          images.append(img)
          #cv2.imwrite('input_eval_img.png', img)
    #            cv2.imshow('img',img)
      except cv2.error as e:
          print('Invalid frame!')
          print(e)



    with tf.Session() as sess:
        ## import and run graph
        tf.import_graph_def(input_graph_def, name = '')

        # get placeholders and tensors
        for i in tf.get_default_graph().get_operations():
            print(i.name)
        x = tf.get_default_graph().get_tensor_by_name(input_node+':0')
        # labels = tf.placeholder(tf.int32, shape = [None, 4])

        # get output tensors
        y = tf.get_default_graph().get_tensor_by_name(output_node+':0')

        sess.run(tf.initializers.global_variables())

        feed_dict ={x: images}
        y_pred = sess.run(y, feed_dict)
        print(y_pred.shape)
        print(y_pred.dtype)
        y_out = np.reshape(y_pred.astype(np.uint8), (256, 512, 4))
        print(y_out.shape)
        print(y_pred)
        image = result_map_to_img(y_pred)
        #image = result_map_to_img(y_out.astype(np.uint8))
        #image = result_map_to_img(y_pred.astype(np.uint8))
        cv2.imwrite('eval_img.png', image)



def main(unused_argv):

    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.gfile.GFile(FLAGS.graph, "rb").read())
    graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)



if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str,
                        default='./trained_models/1210_unet_epoch_21/frozen_graph.pb',
                        help='graph file (.pb) to be evaluated.')
    parser.add_argument('--input_node', type=str,
                        default='input_1',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='conv2d_9/BiasAdd',
                        help='output node.')
    parser.add_argument('--class_num', type=int,
                        default=4,
                        help='number of classes.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
