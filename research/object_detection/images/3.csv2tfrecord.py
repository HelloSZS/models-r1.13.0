# csv2tfrecord.py

# -*- coding: utf-8 -*-


"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python csv2tfrecord.py --csv_input=images/dove_three/train.csv  --output_path=images/train.record
  # Create test data:
  python csv2tfrecord.py --csv_input=images/test_three/test.csv  --output_path=images/test.record
"""


import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from collections import namedtuple, OrderedDict

import sys
sys.path.append("..")
from utils import dataset_util


# os.chdir('/home/ying/usb/models/models-1.12.0/research/object_detection')

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# 原版
def class_text_to_int(row_label):   # 根据自己的标签修改
    if row_label == 'Gold Mineral':
        return 1
    elif row_label == 'Silver Mineral':
        return 2
    else:
        None


# 改版
# TO-DO replace this with label map
list1 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
         'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
         'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'Copper Mineral']

list2 = range(len(list1))
dic = dict(zip(list1,list2))

def class_text_to_int(row_label):   # 根据自己的标签修改
    return dic[row_label] + 1

# print(class_text_to_int('train'))



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    print(width)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        print(row['class'])
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(unuse_argv):
    print(FLAGS.output_path)
    # output_path = './test.record'
    output_path = f'./{FLAGS.output_path}'
    # csv_input = './test.csv'
    csv_input = f'./{FLAGS.csv_input}'

    # print(output_path, csv_input)
    writer = tf.python_io.TFRecordWriter(output_path)
    path = './JPEGImages'         #  需改动
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        print(group)
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))
#
# main()
#
if __name__ == '__main__':
    tf.app.run()