import os
import argparse
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
import numpy

import model as modellib
import config

# Main arguments that can override the config file parameters
argparser = argparse.ArgumentParser(
    description='Export a model following a training')

argparser.add_argument('--model_path',
                       help='Locations of the h5 file containing the trained '
                            'weights. If none is provided, '
                            'the last trained h5 file fill be loaded within log_dirpath',
                       type=str, default= '')

class MyConfig(config.Config):
    NAME = "class"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + nucleus
    TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 5000 # check mask_train for the final value
    VALIDATION_STEPS = 50
    DETECTION_MAX_INSTANCES = 1000
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.35
    RPN_NMS_THRESHOLD = 0.55
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

sess = tf.Session()
K.set_session(sess)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
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
    print("Outputs:")
    print(output_names)

    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def _main_(args):
    # Get the parameters from args
    log_dirpath = "logDir"
    model_path = args.model_path

    # Load the configs in the model folder
    config_inference = MyConfig()
    config_inference.display()

    model = modellib.MaskRCNN(
        mode="inference",
        config=config_inference,
        model_dir=log_dirpath)

    model_path = args.model_path
    model.load_weights(model_path, by_name=True)

    filebase = os.path.splitext(os.path.basename(model_path))[0]
    print('path: '+filebase)
    filename = filebase + ".pb"
    os.makedirs(filebase, exist_ok=True)
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.keras_model.outputs][:4])

    tf.train.write_graph(frozen_graph, filebase, filename, as_text=False)



    anchor_sizes = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]

    for anchor_size in anchor_sizes:
        print('Writing anchors for ' + str(anchor_size))
        anchors = model.get_anchors((anchor_size, anchor_size))
        anc_filepath = os.path.join(filebase, str(anchor_size) + ".anc")
        anc_file = open(anc_filepath, 'w')
        print(anc_filepath)
        for anchor in anchors:
            anc_file.write(str(anchor[0]) + ';' + str(anchor[1]) + ';' + str(anchor[2]) + ';' + str(anchor[3]) + '\r\n')
        anc_file.close()


        anc_file_bin = open(os.path.join(filebase, str(anchor_size) + ".dat"), 'wb')
        all_anchors = []
        for anchor in anchors:
            all_anchors.append(anchor[0])
            all_anchors.append(anchor[1])
            all_anchors.append(anchor[2])
            all_anchors.append(anchor[3])

        float_array = numpy.array(all_anchors, 'float32')
        float_array.tofile(anc_file_bin)
        anc_file_bin.close()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)



