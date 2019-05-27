Convenience utilities for object detection models from [tensorflow model zoo](https://github.com/tensorflow/models/).
See installation and usage instructions below.

# Install TensorFlow Research Models

1. Install the following packages for python 3.5+
```bash
pip3 install Cython
pip3 install contextlib2
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib
pip3 install pycocotools
pip3 install tensorflow-gpu # or CPU version: pip3 install tensorflow
```

2. Download TensorFlow Models:
```bash
git clone https://github.com/tensorflow/models
```

3. Go to <path_to_tf_models>/models/research. Download protoc 3.0.0 using commands below (the latest version of protoc won't work):
```bash
# From <path_to_tf_models>/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

4. Compile *.proto files using downloaded protoc:
```bash
# From <path_to_tf_models>/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

# Add Path to TF Models to .bashrc
Add the following to your .bashrc (replace <path_to_tf_models> with your path):
```bash
export PATH_TO_TF_MODELS_RESEARCH=<path_to_tf_models>/models/research/
export PYTHONPATH="${PYTHONPATH}:$PATH_TO_TF_MODELS_RESEARCH:$PATH_TO_TF_MODELS_RESEARCH/slim"
```

# Downloading and Configuring a Model

1. First and foremost, assign a unique id to each class, and change `label_map.pbtxt` accordingly. Ids should start from 1, not 0 (id: 0 is reserved, and means that nothing was
detected). Here's and example:
```
item {
  id: 1
  name: 'cat'
} 
item {
  id: 2
  name: 'dog'
}
item {
  id: 3
  name: 'frog'
} 
```

2. Choose and download a model from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

3. `pipeline.config` included into each of those models [is no longer valid](https://github.com/tensorflow/models/issues/3794#issuecomment-376972448). Download the latest config
for your model [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs), rename it to `pipeline.config`, and put in into the directory with model checkpoint.

4. Edit your `pipeline.config`. You need to change the following (note that you need to insert `from_detection_checkpoint: true` into `train_config`):
```
model {
  <model_name> {
    # ...
    
    # Specify your number of classes (not including class 0, i.e.
    # the number of classes should be 3 for the above label_map.pbtxt)
    num_classes: 90 
    
    # ...
  }
}
train_config: {
  # ...
  
  # Specify ABSOLUTE path to the downloaded checkpoint. The checkpoint is split into at least 3 files:
  # model.ckpt.index
  # model.ckpt.meta
  # model.ckpt.data-00000-of-00001
  # You need to specify only their common prefix, i.e. /home/user/model/model.ckpt
  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt" 
  # Insert this line here
  from_detection_checkpoint: true
  
  # ...
  
  # Specify the desired batch size. You'll run out of memory if the batch size is too large.
  batch_size: 64
  
  # ...
  
  # Specify the number of training steps. One step = the model is trained on one batch.
  # Recommended number of steps is at least <number_of_images>/<batch_size> * 100
  # (You can't write expressions in this config. You need to calculate the number of steps yourself,
  # and write an integer here.)
  num_steps: 25000
}
train_input_reader: {
  tf_record_input_reader {
    # Specify ABSOLUTE path to the train tfrecord (see below)
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record-00000-of-00100"
  }
  
  # Specify ABSOLUTE path to your label map
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
}
# In tensoflow model zoo, the validation set is called "evaluation" set (or eval for short) 
eval_config: {
  # ...
  
  # This field is deprecated: https://github.com/tensorflow/models/issues/5059#issuecomment-420532746
  # Remove it or comment out.
  num_examples: 8000
}
eval_input_reader: {
  # Specify ABSOLUTE path to the evaluation tfrecord (see below)
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record-00000-of-00010"
  }
  # Specify ABSOLUTE path to your label map (it should be the same as in train_input_reader)
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  
  # ...
}
```

# Data Format and Conversion to TFRecord
## Data Format
1. TFRecord supports only jpg and png. Images of other formats should be converted to jpg or png.
2. Make sure that you described all of your classes in `label_map.pbtxt`, and specified the correct num_classes in `pipeline.config`.
Bounding boxes with classes that are not described in `label_map.pbtxt` will be ignored, and not included into TFRecord.
3. Data should be labeled using [labelimg](https://github.com/tzutalin/labelImg) in Pascal VOC format.
4. You can use one of the two data directory structures:
    1. Put all of your images into `<data_directory>/images`, and all of your labels (`*.xml`) into `<data_directory>/labels`. Then use `convert_and_split_one_folder_to_tfrecord.py` to split it into train, eval (in tensoflow model zoo, the validation set is called "evaluation" set, or eval for short), and test (more details in [Conversion to TFRecord](#conversion-to-tfrecord)).
    2. If your data consists of groups of highly correlated images, and you want to make sure that images from one group won't end up in different sets, then you can make separate directories in your data directory, for example: `<data_directory>/0/images`, `<data_directory>/0/labels`, `<data_directory>/1/images`, `<data_directory>/1/labels`, ..., `<data_directory>/100/images`, `<data_directory>/100/labels` (names can be arbitraty, not necessarily numeric). Images inside one group (for example `<data_directory>/0`) will all go into only one set (e.g. train). Then use `convert_and_split_folders_to_tfrecord.py` (more details in [Conversion to TFRecord](#conversion-to-tfrecord)).

## Conversion to TFRecord
Currently only Pascal VOC format is supported.

Use either `convert_and_split_one_folder_to_tfrecord.py` or `convert_and_split_folders_to_tfrecord.py` depending on your data directory structure (see [Data Format](#data-format)).
Here are a few examples:
```bash
python3 convert_and_split_one_folder_to_tfrecord.py --input_folder=data --output_prefix=records/data --label_map_path=label_map.pbtxt --split=0.9:0.05:0.05
```
```bash
python3 convert_and_split_folders_to_tfrecord.py --input_folder=data --output_prefix=records/data --label_map_path=label_map.pbtxt --split=0.9:0.05:0.05
```
Type `convert_and_split_one_folder_to_tfrecord.py --help` and `convert_and_split_folders_to_tfrecord.py` to get more information about input parameters.

# Training the Model
Type
```bash
./train_and_evaluate_model.sh <model_directory>
```
To start training the model. Checkpoints will be saved into `<model_directory>/train` and `<model_directory>/train/all_checkpoints`.
Tensorboard logs and evaluation results will be saved into `<model_directory>/train`.

# Freezing the Model
Choose the best checkpoint based on evaluation loss:
1. Run `tensorboard --logdir=<model_directory>/train`
2. Look at SCALARS -> Loss (note the capital letter L) -> total_loss, and choose the best checkpoint
3. Run `./freeze_model.sh <model_directory> <checkpoint_number>`
4. Your model will be in `<model_directory>/frozen_model_<date_and_time>/frozen_inference_graph.pb`

Alternatively, you can specify output path:
```bash
./freeze_model.sh <model_directory> <checkpoint_number> <output_path>
```
Then your model will be in `<output_path>/frozen_inference_graph.pb`
