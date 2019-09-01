import os
import shutil
import scipy.misc
import numpy as np
import json

from model import DCGAN
from model import gen_random
#from utils import pp, visualize, to_json, show_all_variables, expand_path, timestamp, get_image
from utils import *

import tensorflow as tf
from keras.preprocessing import image as kimage
import keras

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'



flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
#flags.DEFINE_string("data_dir", "./ref/dcgan/carpedm20--tf/data/", "path to datasets [e.g. $HOME/data]")
#flags.DEFINE_string("data_dir", "/home/wucf20/Documents/home/wucf20/Desktop/TruongHV/2019/age-progression/datasets/UTKFace-clean-13plus", "path to datasets [e.g. $HOME/data]")
flags.DEFINE_string("data_dir", "/home/wucf20/Documents/home/wucf20/Desktop/TruongHV/2019/age-progression/datasets/asian-mixed--w-hair", "path to datasets [e.g. $HOME/data]")
flags.DEFINE_string("out_dir", "./out", "Root directory for outputs [e.g. $HOME/out]")
flags.DEFINE_string("out_name", "", "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Folder (under out_root_dir/out_name) to save samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
# Note that any flag default to False will have the following behaviour:
# When you run main.py w/o specifying the flag, it'll be False;
# when you run main.py and do specify the flag, it'll be True.
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("export", False, "True for exporting with new batch size")
flags.DEFINE_boolean("freeze", False, "True for exporting with new batch size")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
flags.DEFINE_integer("sample_freq", 100, "sample every this many iterations")
flags.DEFINE_integer("ckpt_freq", 200, "save checkpoint every this many iterations")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_string("z_dist", "uniform_signed", "'normal01' or 'uniform_unsigned' or 'uniform_signed'")
flags.DEFINE_boolean("G_img_sum", False, "Save generator image summaries in log")
#flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
  # expand user name and environment variables
  FLAGS.data_dir = expand_path(FLAGS.data_dir)
  FLAGS.out_dir = expand_path(FLAGS.out_dir)
  FLAGS.out_name = expand_path(FLAGS.out_name)
  FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
  FLAGS.sample_dir = expand_path(FLAGS.sample_dir)

  if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
  if FLAGS.input_width is None: FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height

  # output folders
  if FLAGS.out_name == "":
      FLAGS.out_name = '{} - {} - {}'.format(timestamp(), FLAGS.data_dir.split('/')[-1], FLAGS.dataset) # penultimate folder of path
      if FLAGS.train:
        FLAGS.out_name += ' - x{}.z{}.{}.y{}.b{}'.format(FLAGS.input_width, FLAGS.z_dim, FLAGS.z_dist, FLAGS.output_width, FLAGS.batch_size)

  FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
  FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)
  FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)

  if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)

  with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
    flags_dict = {k:FLAGS[k].value for k in FLAGS}
    json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
  

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
          out_dir=FLAGS.out_dir,
          max_to_keep=FLAGS.max_to_keep)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=6,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
          out_dir=FLAGS.out_dir,
          max_to_keep=FLAGS.max_to_keep)
    
    """
    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      load_success, load_counter = dcgan.load(FLAGS.checkpoint_dir)
      if not load_success:
        raise Exception("Checkpoint not found in " + FLAGS.checkpoint_dir)


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
      if FLAGS.export:
        export_dir = os.path.join(FLAGS.checkpoint_dir, 'export_b'+str(FLAGS.batch_size))
        dcgan.save(export_dir, load_counter, ckpt=True, frozen=False)

      if FLAGS.freeze:
        export_dir = os.path.join(FLAGS.checkpoint_dir, 'frozen_b'+str(FLAGS.batch_size))
        dcgan.save(export_dir, load_counter, ckpt=False, frozen=True)

      if FLAGS.visualize:
        OPTION = 1
        visualize(sess, dcgan, FLAGS, OPTION, FLAGS.sample_dir)
    """
    print(bcolors.OKGREEN + bcolors.BOLD, end='')
    print("checkpoint_dir = {}".format(FLAGS.checkpoint_dir))
    print(bcolors.ENDC)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      #print(bcolors.OKGREEN + bcolors.BOLD, end='')
      #print(wucf)
      #print(bcolors.ENDC)
      #ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      #ckpt_name = "model.b64-30000"
      ckpt_name = "model.b64-23000"
      print("ckpt.model_checkpoint_path = {}".format(ckpt.model_checkpoint_path))
      print("ckpt_name = {}".format(ckpt_name))
      dcgan.saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
      #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      counter = int(ckpt_name.split('-')[-1])
      print(bcolors.OKGREEN + bcolors.BOLD, end='')
      print(" [*] Success in reading {}".format(ckpt_name))
      print(bcolors.ENDC)

      # Read/load images here: from z-of-who/
      people = []
      images = []
      work_dir = "z-of-who/"
      for fname in os.listdir(work_dir):
          if os.path.isdir(os.path.join(work_dir, fname)):
              continue
          person = fname.split('.')[0]
          people.append(person)
          person_dir = os.path.join(work_dir, person)
          if not os.path.isdir(person_dir):
              os.mkdir(person_dir)
          image = get_image(os.path.join(work_dir, fname))
          images.append(image)
          old_fpath = os.path.join(work_dir, fname)
          new_fpath = os.path.join(person_dir, fname)
          #os.rename(old_fpath, new_fpath)
          shutil.copy(old_fpath, new_fpath)

      # Not sure if need to convert the list, images, to ndarray.
      images = np.array(images)
      n_people = len(people)

      encoder_dir = "encoder-train-result"
      #encoder_dir = "encoder-train-result-last-bugPlot"
      Ez = keras.models.load_model(os.path.join(encoder_dir, "ckpt_Ez.h5"))
      Ey = keras.models.load_model(os.path.join(encoder_dir, "ckpt_Ey.h5"))

      zs = Ez.predict(images)
      ys = Ey.predict(images)
      print(bcolors.OKGREEN + bcolors.BOLD, end='')
      for i, name in enumerate(people):
        predicted_age = np.argmax(ys[i])
        print("sub = {}, predicted_age = {}".format(name, predicted_age))
      print(bcolors.ENDC)


      A = np.eye(6)
      # A: six age groups
      B = A.copy()
      if n_people > 1:
        for i in range(1, n_people):
          B = np.r_[B, A]
      #B = np.r_[A,A,A,A,A,A,A,A,A,A, np.zeros((4, 6))]
      B = np.r_[B, np.zeros((4, 6))]
      zs_6x = np.zeros((6*n_people, 100))
      print(bcolors.OKGREEN + bcolors.BOLD, end='')
      print("zs_6x.shape = {}, zs.shape = {}".format(zs_6x.shape, zs.shape))
      print(bcolors.ENDC)
      for i in range(n_people):
        zs_6x[6*i: 6*i + 6] = np.repeat(zs[i][np.newaxis, ...], 6, axis=0)
      #sample_10_z = sample_z[:10]
      #sample_10_z = np.repeat(sample_10_z, 6, axis=0)
      #sample_10_z = np.r_[sample_10_z, np.zeros((4, 100))]
      zs_6x = np.r_[zs_6x, np.zeros((4, 100))]
      x_pred = sess.run(dcgan.sampler,
        feed_dict={
            dcgan.z: zs_6x,
            dcgan.y: B,},)
      print(bcolors.OKGREEN + bcolors.BOLD, end='')
      print(x_pred.shape)
      # shape: (n_people*6, 64, 64, 3)
      print(bcolors.ENDC)

      for i, person in enumerate(people):
        save_dir = os.path.join(work_dir, person)
        for age in range(6):
          im = kimage.array_to_img(x_pred[6*i + age], scale=True)
          im.save(os.path.join(save_dir, person + '-' + str(age) + '.png'))




          # We save the first 6 images of the 64 images
          # which corresponds to the first person. And we
          # proceed in the same direction, finishing all 10 persons.
          #im = kimage.array_to_img(samples[6*i + j], scale=True)
          #print("im successfully created")
          #save_dir = 'gan-celebA'
      
          #save_dir = os.path.join(img_dir, "subject-" + str(i).zfill(2))
          #print(bcolors.OKGREEN + bcolors.BOLD, end='')
          #print("save_dir =", save_dir)
          #print(bcolors.ENDC)
          #im = image.array_to_img(samples, scale=True)
          #print("samples.shape =", samples.shape)
          #im.save(os.path.join(img_dir, 'sub-' + str(i).zfill(2) + '-age-' + str(j) + '.png'))
    else:
      print(" [*] Failed to find a checkpoint")
      return False

if __name__ == '__main__':
  #wucf = "wucf20"
  #stu_dir = "photo-STU-members"
  #vinh_path = os.path.join(stu_dir, "vinh.png")
  #vinh = get_image(vinh_path)
  #model_dir = "encoder_train_result"
  #Ez = keras.models.load_model(os.path.join(model_dir, "ckpt_Ez.h5"))
  #z_vinh = Ez.predict(vinh[np.newaxis, ...])
  #z_vinh64 = np.repeat(z_vinh, 64, axis=0)
  #A = np.eye(6)
  #B = np.r_[A,A,A,A,A,A,A,A,A,A, np.zeros((4, 6))]
  tf.app.run()
