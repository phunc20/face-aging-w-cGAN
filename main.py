import os
import json

import scipy.misc
import tensorflow as tf
import numpy as np

from model import DCGAN
from utils import show_all_variables, expand_path


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
#flags.DEFINE_integer("input_height", 200, "The size of image to use (will be center cropped). [108]")
#flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
#flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
#flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("data_dir", "/home/wucf20/Documents/home/wucf20/Desktop/TruongHV/2019/age-progression/datasets/UTKFace-clean", "path to datasets [e.g. $HOME/data]")
#flags.DEFINE_string("out_dir", "./cGAN-ckpts", "Root directory for outputs [e.g. $HOME/out]")
#flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
# Note that any flag default to False will have the following behaviour:
# - When you run main.py w/o specifying the flag, it'll be False;
# - When you run main.py and do specify the flag, it'll be True.
flags.DEFINE_integer("max_to_keep", 1000, "maximum number of checkpoints to keep")
#flags.DEFINE_integer("sample_freq", 100, "sample every this many iterations")
flags.DEFINE_integer("ckpt_freq", 100, "save checkpoint every this many iterations")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_string("z_dist", "normal01", '"normal01" (default) or "uniform_unsigned" or "uniform_signed"')
#flags.DEFINE_boolean("G_img_sum", False, "Save generator image summaries in log")
flags.DEFINE_string("continue_on", None, "The path to the directory of old checkpoints that you want to continue training on, e.g., cGAN-ckpts/g2d3-normal01-2019-10-05_12h38m45s/")
FLAGS = flags.FLAGS

def main(_):
  # expand user name and environment variables
  FLAGS.data_dir = expand_path(FLAGS.data_dir)
  #FLAGS.sample_dir = expand_path(FLAGS.sample_dir)

  #if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
  #if FLAGS.input_width is None: FLAGS.input_width = FLAGS.input_height
  #if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height


  config = tf.ConfigProto()
  # One may choose to uncomment one of the following lines
  # and tweak the related value as they wish.
  #config.gpu_options.per_process_gpu_memory_fraction = 0.5
  #config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    dcgan = DCGAN(
        sess,
        #input_width=FLAGS.input_width,
        #input_height=FLAGS.input_height,
        #output_width=FLAGS.output_width,
        #output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=6,
        z_dim=FLAGS.z_dim,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        #crop=FLAGS.crop,
        data_dir=FLAGS.data_dir,
        continue_on=FLAGS.continue_on,
        max_to_keep=FLAGS.max_to_keep)

    #show_all_variables()
    dcgan.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
