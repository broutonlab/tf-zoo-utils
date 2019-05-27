import tensorflow as tf
import xml.etree.ElementTree as ET
import PIL.Image
import os
import io
import re
import random
from tqdm import tqdm

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from convert_common import create_tf_example

def main(_):
    if re.match(r'^[0-9.]+:[0-9.]+:[0-9.]+$', FLAGS.split) is None:
        print('Error: incorrect format of --split')
        return
    train_split, eval_split, test_split = [float(s) for s in FLAGS.split.split(':')]
    if abs(train_split + eval_split + test_split - 1) >= 1e-5:
        print('Error: --split doesn\'t sum up to 1')
        return
    
    class_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    image_directory = os.path.join(FLAGS.input_folder, 'images')
    xml_directory = os.path.join(FLAGS.input_folder, 'labels')
    xml_file_names = os.listdir(xml_directory)
    
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
    random.shuffle(xml_file_names)
    
    image_count = len(xml_file_names)
    eval_count = int(image_count * eval_split)
    test_count = int(image_count * test_split)
    train_count = image_count - eval_count - test_count
    
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
        
    train_xml_file_names = xml_file_names[:train_count]
    eval_xml_file_names = xml_file_names[train_count:(train_count+eval_count)]
    test_xml_file_names = xml_file_names[(train_count+eval_count):]
    
    output_folder = os.sep.join(re.split(r'/|\\', FLAGS.output_prefix)[:-1])
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    set_names = ['train', 'eval', 'test']
    sets = [train_xml_file_names, eval_xml_file_names, test_xml_file_names]
    for set_name, xml_file_names in zip(set_names, sets):
        print('{}:'.format(set_name))
        with tf.python_io.TFRecordWriter('{}_{}.record'.format(FLAGS.output_prefix, set_name)) as record_writer:
            for xml_file_name in tqdm(xml_file_names):
                try:
                    tf_example = create_tf_example(os.path.join(xml_directory, xml_file_name), image_directory, class_dict)
                    record_writer.write(tf_example.SerializeToString())
                except Exception as e:
                    print('Skipping {}: {}'.format(xml_file_name, str(e)))

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_folder', None, 'Path to the data folder')
    flags.DEFINE_string('output_prefix', 'data', 'Path and prefix to resulting records. For example, if --output_prefix=records/data, then '
                                                 'the names will be records/data_train.record, records/data_eval.record, and records/data_test.record')
    flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map')
    flags.DEFINE_string('split', '0.8:0.1:0.1', 'Split for train:eval:test')
    flags.DEFINE_integer('seed', None, 'Random generator seed for splitting')
    flags.DEFINE_boolean('yes', False, 'Don\'t ask before splitting')
    FLAGS = flags.FLAGS
    tf.app.run()
