import tensorflow as tf
import os
import sys
from tqdm import tqdm
from convert_common import *

from object_detection.utils import label_map_util

def main(_):
    class_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    folder_names, image_counts = ls_data_folder(FLAGS.input_folder)
    
    with tf.python_io.TFRecordWriter(FLAGS.output_path) as record_writer:
        for folder_name in tqdm(folder_names):
            folder_to_record(record_writer, os.path.join(folder_name, 'labels'), os.path.join(folder_name, 'images'), class_dict)
                

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_folder', None, 'Path to the top folder')
    flags.DEFINE_string('output_path', None, 'Path to the output record.')
    flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map')
    FLAGS = flags.FLAGS
    tf.app.run()
