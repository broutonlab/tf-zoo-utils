import tensorflow as tf
import os
import random
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_string('input_path', 'data.record', 'Path to input TFRecord')
flags.DEFINE_string('output_prefix', 'data', 'Prefix of output TFRecord files. For example, if output_prefix is \'folder/data\', then the data will be saved to ' +
                                             '\'folder/data_train.record\', \'folder/data_eval.record\', and \'folder/data_test.record\'.')
flags.DEFINE_string('data_split', '0.7:0.15:0.15', 'The data split in the following format: <train_split>:<eval_split>:<test_split>.')
flags.DEFINE_integer('random_seed', 42, 'Random seed for splitting. If random_seed is -1 then system time is used.')
FLAGS = flags.FLAGS


def main(_):
    splits = [float(x) for x in FLAGS.data_split.split(':')]
    
    if abs(sum(splits) - 1.0) > 1e-8:
        print('data_split should sum up to 1')
        return
    
    full_dataset_size = 0
    for record in tf.python_io.tf_record_iterator(FLAGS.input_path):
        full_dataset_size += 1
        
    eval_size = int(splits[1] * full_dataset_size)
    test_size = int(splits[2] * full_dataset_size)
    
    if FLAGS.random_seed == -1:
        random.seed(None)
    else:
        random.seed(FLAGS.random_seed)
    indices = random.sample(list(range(full_dataset_size)), eval_size+test_size)
    
    eval_indices = set(indices[0:eval_size])
    test_indices = set(indices[eval_size:])
    
    print('Splitting data:')
    
    with tf.python_io.TFRecordWriter(FLAGS.output_prefix + '_train.record') as train_writer, \
         tf.python_io.TFRecordWriter(FLAGS.output_prefix + '_eval.record') as eval_writer, \
         tf.python_io.TFRecordWriter(FLAGS.output_prefix + '_test.record') as test_writer:
        i = 0
        for record in tqdm(tf.python_io.tf_record_iterator(FLAGS.input_path), total=full_dataset_size):
            if i in eval_indices:
                eval_writer.write(record)
            elif i in test_indices:
                test_writer.write(record)
            else:
                train_writer.write(record)
            i += 1
    print('Done')
    print('train size:', full_dataset_size - eval_size - test_size)
    print('eval size:', test_size)
    print('test size:', eval_size)

if __name__ == '__main__':
  tf.app.run()
