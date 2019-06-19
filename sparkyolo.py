from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import argparse
import tensorflow as tf
import cv2
from tensorflowonspark import TFCluster
import config
import yolo_dist
from data_aug import *
import logging

conf = SparkConf().setAppName('YOLOv3')
sc = SparkContext(conf=conf)
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

print('-------------')
print('-----------------------------------------')
print('-----------------------------------')
print(executors)
print('--------------------')

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("-o", "--output", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("--cluster_size", help="number of epochs", type=int, default=4)
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("--rdma", help="use rdma connection", default=False)
parser.add_argument("--mode", help="use rdma connection", default="train")

args = parser.parse_args()
print(args)

input_data = sc.newAPIHadoopFile(args.input, "org.tensorflow.hadoop.io.TFRecordFileInputFormat"
                                 , keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable")


def _parse_record(bytestr, class_num, img_size, anchors, mode):
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    features = example.features.feature
    labels = np.array(features['labels'].int64_list.value)
    shape = np.array(features['shape'].int64_list.value)
    img = np.array(features['image'].int64_list.value).reshape(shape)
    boxes = np.array(features['boxes'].int64_list.value).reshape([-1, 4])
    img_idx = np.array(features['index'].int64_list.value)[0]
    img = img.astype(np.float32)
    boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32
                                           )), axis=-1)
    if mode == 'train':
        # random color jittering
        # random color jittering
        # NOTE: applying color distort may lead to bad performance sometimes
        # img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 2:
            img, boxes = random_expand(img, boxes, 2)

        # random cropping
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0 + h, x0: x0 + w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


print("=======================================", input_data.count())
dataRDD = input_data.map(
    lambda x: _parse_record(bytes(x[0]), config.class_num, config.img_size, config.anchors, 'train'))
# To be modified
cluster = TFCluster.run(sc, yolo_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                        TFCluster.InputMode.SPARK, log_dir=config.log_dir)
if args.mode == "train":
    print("======================================================Training======================")
    logging.info(args.epochs)
    cluster.train(input_data, args.epochs)
cluster.shutdown()
