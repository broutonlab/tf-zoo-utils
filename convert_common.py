import tensorflow as tf
import xml.etree.ElementTree as ET
import PIL.Image
import os
import io
import warnings

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


class UnknownLabelWarning(Warning):
    pass

def ls_data_folder(data_folder):
    folder_names = os.listdir(data_folder)
    folder_names.sort()
    folder_names = [os.path.join(data_folder, folder_name) for folder_name in folder_names]
    folder_names = [folder_name for folder_name in folder_names if os.path.isdir(folder_name)]
    image_counts = [len(os.listdir(os.path.join(folder_name, 'labels'))) for folder_name in folder_names]
    return folder_names, image_counts

def folder_to_record(record_writer, xml_folder, image_folder, class_dict):
    for file_name_xml in os.listdir(xml_folder):
        try:
            tf_example = create_tf_example(os.path.join(xml_folder, file_name_xml), image_folder, class_dict)
            record_writer.write(tf_example.SerializeToString())
        except Exception as e:
            print('Skipping {}: {}'.format(file_name_xml, str(e)))

def create_tf_example(full_path_xml, full_path_images, class_dict):
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    xml_tree = ET.parse(full_path_xml)
    xml_root = xml_tree.getroot()

    path_from_xml = xml_root.find('path').text.replace('\\', '/')
    full_path_image = os.path.join(full_path_images, path_from_xml.split('/')[-1])

    with tf.gfile.GFile(full_path_image, 'rb') as fid:
        encoded_image_data = fid.read()
    encoded_image_io = io.BytesIO(encoded_image_data)
    image = PIL.Image.open(encoded_image_io)
    image.verify()
    
    if image.format == 'JPEG':
        image_format = b'jpeg'
    elif image.format == 'PNG':
        image_format = b'png'
    else:
        raise ValueError('Invalid image format: {} (expeced JPEG or PNG)'.format(image.format))

    width, height = image.size
    
    for xml_object in xml_root.findall('object'):
        xml_object_class = xml_object.find('name').text
        if xml_object_class in class_dict:
            xml_bndbox = xml_object.find('bndbox')
            xmins.append(float(xml_bndbox.find('xmin').text) / width)
            xmaxs.append(float(xml_bndbox.find('xmax').text) / width)
            ymins.append(float(xml_bndbox.find('ymin').text) / height)
            ymaxs.append(float(xml_bndbox.find('ymax').text) / height)
            classes_text.append(xml_object_class.encode('utf-8'))
            classes.append(class_dict[xml_object_class])
        else:
            warning_message = 'Label \'{}\' is not in the label map. Objects with this label will not be included into the resulting tfrecord.'.format(xml_object_class)
            warnings.warn(warning_message, UnknownLabelWarning)
    

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(full_path_image.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(full_path_image.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
  
if __name__ == '__main__':
    print('This is a module with convert functions. Do not run it.')
