from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import cv2
import tensorflow as tf
import tensorflowonspark
import config
import yolo_dist
from data_aug import *

conf = SparkConf().setAppName('ckpt_test')
sc = SparkContext(conf=conf)
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="HDFS path to MNIST images in parallelized format")
args_shell = parser.parse_args()
print(args_shell)



print('-------------')
print('-----------------------------------------')
print('-----------------------------------')
print(executors)
print('--------------------')

##################
# Model definition
##################
yolo_model = yolov3(config.class_num, config.anchors, config.use_label_smooth, config.use_focal_loss, config.batch_norm_decay,
                    config.weight_decay)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=config.restore_part))
update_vars = tf.contrib.framework.get_variables_to_restore(include=config.update_part)
# summary part
tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])



global_step = tf.Variable(float(config.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
if config.use_warm_up:
    learning_rate = tf.cond(tf.less(global_step, config.train_batch_num * config.warm_up_epoch),
                            lambda: config.learning_rate_init * global_step / (
                                    config.train_batch_num * config.warm_up_epoch),
                            lambda: config_learning_rate(config,
                                                         global_step - config.train_batch_num * config.warm_up_epoch))
else:
    learning_rate = config_learning_rate(config, global_step)
tf.summary.scalar('learning_rate', learning_rate)

if not config.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

optimizer = config_optimizer(config.optimizer_name, learning_rate)

if config.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
merged = tf.summary.merge_all()
glo_init=tf.global_variables_initializer()
loc_init=tf.local_variables_initializer()

hooks = [tf.train.StopAtStepHook(last_step=1000000)]

logdir = ctx.absolute_path(config.log_dir)
if job_name == "worker" and task_index == 0:
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

with tf.train.MonitoredTrainingSession(master=server.target,
                                       is_chief=(task_index == 0),
                                       scaffold=tf.train.Scaffold(
                                           init_op=glo_init,
                                           local_init_op=loc_init,
                                           summary_op=merged,
                                           saver=saver_best),
                                       checkpoint_dir=logdir,
                                       hooks=hooks) as sess:

    saver_to_restore.restore(sess, config.restore_path)

    print('\n----------- start to train -----------\n')

    best_mAP = -np.Inf

    tf_feed = ctx.get_data_feed(config.mode == "train")
    while not sess.should_stop() and not tf_feed.should_stop():
        loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        image, y_true = feed_dict(tf_feed.next_batch(config.batch_size))
        feed = {image:image, y_true:y_true, is_training:True}
        while len(image) > 0:
            _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                feed_dict=feed)

            writer.add_summary(summary, global_step=__global_step)

            loss_total.update(__loss[0], len(__y_pred[0]))
            loss_xy.update(__loss[1], len(__y_pred[0]))
            loss_wh.update(__loss[2], len(__y_pred[0]))
            loss_conf.update(__loss[3], len(__y_pred[0]))
            loss_class.update(__loss[4], len(__y_pred[0]))

            if __global_step % config.train_evaluation_step == 0 and __global_step > 0:
                # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.eval_threshold)
                recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred,
                                                    __y_true, config.class_num, config.eval_threshold)

                if task_index == 0:
                    writer.add_summary(make_summary('evaluation/train_batch_precision', precision),
                                       global_step=__global_step)

                if np.isnan(loss_total.average):
                    print('****' * 10)
                    raise ArithmeticError(
                        'Gradient exploded! Please train again and you may need modify some parameters.')

        tmp_total_loss = loss_total.average
        loss_total.reset()
        loss_xy.reset()
        loss_wh.reset()
        loss_conf.reset()
        loss_class.reset()
