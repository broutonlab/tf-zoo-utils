import tensorflow as tf
import xml.etree.ElementTree as ET
import PIL.Image
import os
import io
from tqdm import tqdm

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from convert_common import create_tf_example

def main(_):
    xml_directory = os.path.join(FLAGS.data_path, 'labels')
    class_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for file_name_xml in tqdm(os.listdir(xml_directory)):
        try:
            tf_example = create_tf_example(os.path.join(xml_directory, file_name_xml), os.path.join(FLAGS.data_path, 'images'), class_dict)
            writer.write(tf_example.SerializeToString())
        except Exception as e:
            print('Skipping {}: {}'.format(file_name_xml, str(e)))

    writer.close()


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('output_path', 'data.record', 'Path to output TFRecord')
    flags.DEFINE_string('data_path', 'data', 'Path to data')
    flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map')
    FLAGS = flags.FLAGS
    tf.app.run()
