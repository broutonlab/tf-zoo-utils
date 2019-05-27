import tensorflow as tf
import os
import random
import re
import sys
from tqdm import tqdm
from convert_common import *

from object_detection.utils import label_map_util

def main(_):
    if re.match(r'^[0-9.]+:[0-9.]+:[0-9.]+$', FLAGS.split) is None:
        print('Error: incorrect format of --split')
        return
    train_split, eval_split, test_split = [float(s) for s in FLAGS.split.split(':')]
    if abs(train_split + eval_split + test_split - 1) >= 1e-5:
        print('Error: --split doesn\'t sum up to 1')
        return
    
    class_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    all_folder_names, image_counts = ls_data_folder(FLAGS.input_folder)
    
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
    indices = list(range(len(all_folder_names)))
    random.shuffle(indices)
    all_folder_names = [all_folder_names[i] for i in indices]
    image_counts = [image_counts[i] for i in indices]
    
    # Obtain desired splits
    image_count = sum(image_counts)
    eval_desired_count = int(image_count * eval_split)
    test_desired_count = int(image_count * test_split)
    
    # Calculate actual splits (images from one folder must not go into 2 different sets)
    i = 0
    test_count = 0
    test_folder_names = []
    while test_count < test_desired_count:
        test_count += image_counts[i]
        test_folder_names.append(all_folder_names[i])
        i += 1
    
    eval_count = 0
    eval_folder_names = []
    while eval_count < eval_desired_count:
        eval_count += image_counts[i]
        eval_folder_names.append(all_folder_names[i])
        i += 1
        
    train_count = 0
    train_folder_names = []
    while i < len(all_folder_names):
        train_count += image_counts[i]
        train_folder_names.append(all_folder_names[i])
        i += 1
        
    print('The resulting split:')
    print('train:', train_count)
    print('eval: ', eval_count)
    print('test: ', test_count)
    
    if not FLAGS.yes:
        answer = None
        while answer not in ['y', 'n', 'Y', 'N']:
            answer = input('Continue? (y|n) ')
        if answer in ['n', 'N']:
            return
    
    output_folder = os.sep.join(re.split(r'/|\\', FLAGS.output_prefix)[:-1])
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    set_names = ['train', 'eval', 'test']
    sets = [train_folder_names, eval_folder_names, test_folder_names]
    for set_name, folder_names in zip(set_names, sets):
        print('{}:'.format(set_name))
        with tf.python_io.TFRecordWriter('{}_{}.record'.format(FLAGS.output_prefix, set_name)) as record_writer:
            for folder_name in tqdm(folder_names):
                folder_to_record(record_writer, os.path.join(folder_name, 'labels'), os.path.join(folder_name, 'images'), class_dict)
                

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_folder', None, 'Path to the top folder')
    flags.DEFINE_string('output_prefix', 'data', 'Path and prefix to resulting records. For example, if --output_prefix=records/data, then '
                                                 'the names will be records/data_train.record, records/data_eval.record, and records/data_test.record')
    flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map')
    flags.DEFINE_string('split', '0.8:0.1:0.1', 'Split for train:eval:test')
    flags.DEFINE_integer('seed', None, 'Random generator seed for splitting')
    flags.DEFINE_boolean('yes', False, 'Don\'t ask before splitting')
    FLAGS = flags.FLAGS
    tf.app.run()
