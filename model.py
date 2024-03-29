from __future__ import division
from __future__ import print_function
from glob import glob
import math
import os
from six.moves import xrange
import time

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as kimage

from ops import *
from utils import *
import utils


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def gen_random(mode, size):
  if mode=='normal01': return np.random.normal(0,1,size=size)
  if mode=='uniform_signed': return np.random.uniform(-1,1,size=size)
  if mode=='uniform_unsigned': return np.random.uniform(0,1,size=size)


class DCGAN(object):
  def __init__(self, sess,
    #input_height=108, input_width=108,
    batch_size=64,
    sample_num = 64,
    #output_height=64, output_width=64,
    y_dim=6, z_dim=100, gf_dim=64, df_dim=64,
    gfc_dim=1024, dfc_dim=1024, c_dim=3,
    dataset_name='default',
    max_to_keep=1,
    input_fname_pattern='*.jpg',
    continue_on=None,
    data_dir=os.path.join(os.environ["HOME"],"datasets", "UTKFace")):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]

    Understanding:
      gf_dim: default to 64. Better understood as n_channels.
    """
    self.sess = sess

    self.batch_size = batch_size
    self.sample_num = sample_num

    #self.input_height = input_height
    #self.input_width = input_width
    #self.output_height = output_height
    #self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : 
    # deals with poor initialization, helps gradient flow.
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern  # e.g. *.jpg
    ckpts_dir = "cGAN-ckpts"
    os.makedirs(ckpts_dir, exist_ok=True)
    #self.out_dir = os.path.join(ckpts_dir, "g{}d{}")
    #self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir
    self.continue_on = continue_on
    self.max_to_keep = max_to_keep

    #data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
    data_fpaths = os.path.join(self.data_dir, self.input_fname_pattern)
    self.data = glob(data_fpaths)
    #self.data = glob(os.path.join(
    #    "/home/phunc20/datasets/faces/all-races", "*.jpg"))
    
    if len(self.data) == 0:
      raise Exception("[!] No data found in '" + data_path + "'")
    #np.random.shuffle(self.data)
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    if len(self.data) < self.batch_size:
      raise Exception("[!] Entire dataset size ({}) is less than batch_size ({})".format(len(self.data), self.batch_size))
    
    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
    else:
      self.y = None

    image_dims =  [*utils.IMG_RESOLUTION, self.c_dim]
    # Normally, we should have
    # image_dims equal to [64, 64, 3]

    self.inputs = tf.placeholder(
      tf.float32, [None] + image_dims, name='real_images')

    self.inputs2 = tf.placeholder(
      tf.float32, [None] + image_dims, name='real_img_wrong_age')

    inputs = self.inputs
    inputs2 = self.inputs2

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G = self.generator(self.z, self.y)
    self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
    # self.y2 will consist of wrong ages, e.g. young face w/ old age.
    self.y2 = tf.placeholder(tf.float32, [None, self.y_dim], name='y2')
    self.D2, self.D_logits2 = self.discriminator(inputs2, self.y2, reuse=True)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #self.d_loss_real = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    #self.d_loss_real = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits, 0.9*tf.ones_like(self.D)))

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits,
        tf.random.uniform(tf.shape(self.D), minval=0.8, maxval=1),
        ))
   

    #self.d_loss_fake = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    #self.d_loss_fake = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits_, 0.1*tf.ones_like(self.D_)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_,
        tf.random.uniform(tf.shape(self.D_), minval=0, maxval=0.2),
        ))
   

    #self.d_loss_real_wrong = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits2, tf.zeros_like(self.D2)))
    #self.d_loss_real_wrong = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits2, 0.03*tf.ones_like(self.D2)))
    self.d_loss_real_wrong = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits2,
        tf.random.uniform(tf.shape(self.D2), minval=0, maxval=0.2),
        ))


    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    #self.g_loss = tf.reduce_mean(
    #  sigmoid_cross_entropy_with_logits(self.D_logits_, 0.9*tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_real_wrong
    #self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

  def train(self, config):
    """
    In main.py, we shall call this method with
      train(FLAGS)
    So, think of config as FLAGS.
    """
    # Optimizers' choice
    # N.B. This part of initialization of var should
    # DEFINITELY lie before saver.restore()!!
    # You have suffered from this but a lot.
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    
    # Construct (or Identify the existing)
    # folders for ckpts and image files.
    cGAN_img_eval = "cGAN-img-eval"
    os.makedirs(cGAN_img_eval, exist_ok=True)
    n_times_g = 2
    n_times_d = 3
    if not self.continue_on:
      print("continue_on is NOT set")
      tstamp = "g{}d{}-".format(n_times_g, n_times_d) \
                + config.z_dist + "-" + \
                time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
      print("tstamp = {}".format(tstamp))
      input("Arretez ce programme par (Ctrl + C) si tu veux. Epuyez d'autre key pour continuer.")
    else:
      print("continue_on is set to {}".format(self.continue_on))
      tstamp = self.continue_on.strip(os.sep).split(os.sep)[-1]
      if tstamp == '':
        raise ValueError("Invalid continue_on arg (resulting in an empty string tstamp)")
      print("tstamp = {}".format(tstamp))
      input("Arretez ce programme. (Ctrl + C)")

    #tstamp = "g{}d{}-".format(n_times_g, n_times_d) \
    #          + config.z_dist + "-" + \
    #          time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    img_dir = os.path.join(cGAN_img_eval, tstamp)
    log_file_path = os.path.join(img_dir, "train-log")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(10):
      os.makedirs(
        os.path.join(img_dir, "subject-" + str(i).zfill(2)),
        exist_ok=True,
      )
    
    out_parent_dir = "cGAN-ckpts"
    self.out_dir = os.path.join(out_parent_dir, tstamp)
    # TODO: consider delete next redundant line
    #os.makedirs(self.out_dir, exist_ok=True)
    self.ckpts_dir = os.path.join(self.out_dir, "ckpts")
    #self.ckpts_dir = os.path.join(self.out_dir, "checkpoint")
    os.makedirs(self.ckpts_dir, exist_ok=True)

    counter = 1
    # counter = #(total batches we will run through in config.epoch epochs)
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.ckpts_dir)
    npy_saved_sample_z = "sample_z.npy"
    sample_z_path = os.path.join(self.out_dir, npy_saved_sample_z)
    if could_load:
      counter = checkpoint_counter + 1
      print(" [*] Load ckpt SUCCESSfully")
      if os.path.exists(sample_z_path):
        sample_z = np.load(sample_z_path)
        print("Load sample_z SUCCESSfully!")
      else:
        print(" [!] Intermediate changing period")
        print(" [!] ckpt exists but no sample_z.py")
        np.random.seed(42)
        sample_z = gen_random(config.z_dist, size=(30, self.z_dim)).astype(np.float32)
        np.save(sample_z_path, sample_z)
      #self.load(self.ckpts_dir)
    else:
      print(" [!] Loading failed. Please specify existing checkpoints.")
      #self.build_model()
      print(" [*] Creating 10 new subjects (sample_z)") 
      sample_z = gen_random(config.z_dist, size=(30, self.z_dim)).astype(np.float32)
      np.save(sample_z_path, sample_z)
      
    # sample_z contains more than 10 instances, just to be more flexible.
    # Here I set it to be 30, but it can be any int (>= 10).
    #sample_z = gen_random(config.z_dist, size=(30, self.z_dim)).astype(np.float32)
    #np.save(sample_z_path, sample_z)



    def wrong(y):
      """
      input:
          y, ndarray w/ shape = (?, 6)
            each row of y being a one-hot vector.

      output:
          yy, ndarray w/ the same shape but wrong one-hot rows.
      """
      yy = np.zeros_like(y)
      for i in range(y.shape[0]):
        # Find the non-zero entry in the row y[i]
        idx = 0
        while True:
          if y[i][idx] != 0:
            break
          else:
            idx += 1
        if idx < 3:
          yy[i][np.random.randint(3,5+1)] = 1
        else:
          yy[i][np.random.randint(0,2+1)] = 1
      return yy

    sample_files = self.data[0:self.sample_num]
    sample = [get_image(sample_file) for sample_file in sample_files]
    # dtype conversion to float32 (the default of tensorflow)
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  

    # Construction of sample_10_z and B. (cf. line 524)
    # 7 subjects and 3 random
    sample_05_z = sample_z[:5]
    def random_05_z():
      return gen_random(config.z_dist, size=(5, self.z_dim)).astype(np.float32)
    #sample_10_z = sample_z[:10]
    A = np.eye(6)
    #B = np.r_[A,A,A,A,A,A,A,A,A,A, np.zeros((4, 6))]
    B = np.r_[A,A,A,A,A,A,A,A,A,A]


    def random_batch_real_images():
      #batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
      random_indices = np.random.choice(len(self.data),
                                        config.batch_size,
                                        replace=False,
                                        )
      # or equivalently
      #random_indices = np.random.permutation(len(self.data))\
      #                 [:config.batch_size]
      #batch_files = self.data[random_indices]
      batch_files = [ self.data[i] for i in random_indices ]
      #batch = [get_image(batch_file) for batch_file in batch_files]
      batch = []
      batch_y = []
      batch_fnames = []
      for batch_file in batch_files:
        batch.append(get_image(batch_file))
        #age = int(os.path.basename(batch_file).split('_')[0])
        try:
          age = int(os.path.basename(batch_file).split('_')[0])
        except:
          print(bcolors.OKGREEN + bcolors.BOLD, end='')
          print("batch_file.split('_')[0] = {}".format(batch_file.split('_')[0]))
          print(bcolors.ENDC)
          raise Exception
        fname = os.path.basename(batch_file)
        batch_y.append(age_to_1hot(age))
        batch_fnames.append(fname)
      batch_y = np.array(batch_y).astype(np.float32)
      # Note that we only need batch_y to have shape (64, 6) not (128, 6).
      # This is because the same batch_y is passed separately both to
      # discriminator and gan.
      batch_images = np.array(batch).astype(np.float32)
      
      return batch_images, batch_y, batch_fnames


    for epoch in xrange(config.epoch):
      #self.data = glob(os.path.join(
      #  "/home/wucf20/Documents/home/wucf20/Desktop/TruongHV/2019/age-progression/datasets/UTKFace-clean", "*.jpg"))
      #np.random.shuffle(self.data)
      # Note that each epoch data is shuffled.
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, int(batch_idxs)):
        batch_start_time = time.time()
        
        ## ==========================================================
        ## The old way (deprecated)
        ## ------------------------
        ## in which we take a fixed batch of real images for our
        ## discriminator to train. Cf. below in the for loop of
        ## the discriminator to see the new, random batch.
        ## ==========================================================
        #batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        ##batch = [get_image(batch_file) for batch_file in batch_files]
        #batch = []
        #batch_y = []
        #batch_fnames = []
        #for batch_file in batch_files:
        #  batch.append(get_image(batch_file))
        #  #print(bcolors.OKGREEN + bcolors.BOLD, end='')
        #  #print("batch_file.split('_')[0] = {}".format(batch_file.split('_')[0]))
        #  #print(bcolors.ENDC)
        #  age = int(os.path.basename(batch_file).split('_')[0])
        #  fname = os.path.basename(batch_file)
        #  batch_y.append(age_to_1hot(age))
        #  batch_fnames.append(fname)
        #batch_y = np.array(batch_y).astype(np.float32)
        ## Note that we only need batch_y to have shape (64, 6) not (128, 6).
        ## This is because the same batch_y is passed separately both to
        ## discriminator and gan.
        #if self.grayscale:
        #  batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        #  # grey image shape e.g. (64, 64) instead of (64, 64, 1)
        #else:
        #  batch_images = np.array(batch).astype(np.float32)


        # Update D network
        for _ in range(n_times_d):
          batch_z = gen_random(config.z_dist, size=[config.batch_size, self.z_dim]).astype(np.float32)

          # TODO: Insert here a random batch of real images batch_images
          batch_images, batch_y, _ = random_batch_real_images()
          batch_images2, batch_y2, _ = random_batch_real_images()


          _ = self.sess.run(d_optim,
            feed_dict={self.inputs: batch_images,
                       self.inputs2: batch_images2,
                       self.z: batch_z,
                       self.y: batch_y,
                       self.y2: wrong(batch_y2),
                       })
          #_ = self.sess.run(d_optim,
          #  feed_dict={self.inputs: batch_images,
          #             self.z: batch_z,
          #             self.y: batch_y,
          #             })
        #self.writer.add_summary(summary_str, counter)


        # Update G network
        #_, summary_str = self.sess.run([g_optim, self.g_sum],
        #  feed_dict={self.z: batch_z, self.y: batch_y})

        # N.B. We deliberately sample a new set of batch_z for generator.
        for _ in range(n_times_g):
          batch_z = gen_random(config.z_dist, size=[config.batch_size, self.z_dim]).astype(np.float32)
          _ = self.sess.run(g_optim,
            feed_dict={self.z: batch_z, self.y: batch_y})
        #self.writer.add_summary(summary_str, counter)
        
        ## stdout log for loss, precision, etc.
        ## moved elsewhere. If put here will print everty iteration/
        ## step. Too many to be useful.
        #errD_real, errD_fake, errG, D, D_ = self.sess.run(
        #  [self.d_loss_real, self.d_loss_fake,
        #  self.g_loss, self.D, self.D_], feed_dict={self.z: batch_z,
        #  self.y: batch_y, self.inputs: batch_images,
        #})

        #TP = sum(D > 0.5)
        #FN = sum(D <= 0.5)
        #FP = sum(D_ > 0.5)
        #recall = TP/(TP + FN)
        #precision = TP/(TP + FP)

        #print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
        #  % (counter, epoch, config.epoch, idx, batch_idxs,
        #    time.time() - start_time, errD_fake+errD_real, errG))
        #np.set_printoptions(precision=2, suppress=True)
        ##print("discriminate_for_real = {}".format(D))
        ##print("discriminate_for_fake = {}".format(D_))
        #print("discriminate_for_real       = {:.2f}%".format(100*D.mean()))
        ##print("discriminate_for_real_wrong = {:.2f}%".format(100*D2.mean()))
        #print("discriminate_for_fake       = {:.2f}%".format(100*D_.mean()))
        #print("recall = {:.2f}".format(recall[0]))
        #print("precision = {:.2f}".format(precision[0]))
        #batch_end_time = time.time()
        #ETA = (batch_end_time - batch_start_time)*(int(batch_idxs)*(config.epoch - epoch - 1) + (int(batch_idxs) - idx - 1))
        #hh = int(ETA//(60**2))
        #mm = int(ETA%(60**2)//60)
        #ss = int(ETA%60)
        #print("ETA: {} h {} m {} s".format(hh, mm, ss))
        #print()

        #if np.mod(counter, config.sample_freq) == 0:
        if True:
          # sample_freq == 100 by default
          try:
            sample_10_z = np.r_[sample_05_z, random_05_z()]
            sample_10_z = np.repeat(sample_10_z, 6, axis=0)
            samples = self.sess.run(self.sampler,
              feed_dict={
                  self.z: sample_10_z,
                  self.y: B,
              },
            )
            #print("type(samples) =", type(samples))
            #print("samples.shape =", samples.shape)
            # samples.shape = (None, 64, 64, 3)
            #save_images(samples, image_manifold_size(samples.shape[0]),
            #      './{}/train_{:08d}.png'.format(config.sample_dir, counter))

            # The following two for's are equivalent.
            # The pt is that we do not want i to touch/become 60 one day.
            #for i in range(0, len(sample_10_z) - len(sample_10_z)%6, 6):
            for i in range(10):
              for j in range(6):
                # We save the first 6 images of the 64 images
                # which corresponds to the first person. And we
                # proceed in the same direction, finishing all 10 persons.
                im = kimage.array_to_img(samples[6*i + j], scale=True)
                #print("im successfully created")
                #save_dir = 'gan-celebA'

                #save_dir = os.path.join("gan-celebA", "subject-" + str(i).zfill(2))
                save_dir = os.path.join(img_dir, "subject-" + str(i).zfill(2))
                #print(bcolors.OKGREEN + bcolors.BOLD, end='')
                #print("save_dir =", save_dir)
                #print(bcolors.ENDC)
                #im = image.array_to_img(samples, scale=True)
                #print("samples.shape =", samples.shape)
                #im.save(os.path.join(save_dir, 'sub-' + str(i).zfill(2) + '-step' + '-' + str(counter).zfill(6) + '-age-' + str(j) + '.png'))
                im.save(os.path.join(save_dir, 'step-' +  str(counter).zfill(6) + '-sub-' + str(i).zfill(2) + '-age-' + str(j) + '.png'))
            #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            #dir_for_real = os.path.join("gan-celebA", "real-images")
            #im = kimage.array_to_img(batch_images[0], scale=True)
            #im.save(os.path.join(dir_for_real, batch_fnames[0]))
          except Exception as ex:
            #print(bcolors.OKGREEN + bcolors.BOLD, end='')
            print("one pic error!...")
            print(str(ex.args))
            #print(bcolors.ENDC)

        #if np.mod(counter, config.ckpt_freq) == 0:
        if True:
          # ckpt_freq == 200 by default
          #self.save(config.checkpoint_dir, counter, global_step=counter)
          #self.save(config.checkpoint_dir, counter)
          self.save(self.ckpts_dir, counter)

          # Stdout log for loss, precision, recall, etc.
          errD_real, errD_fake, errG, D, D_ = self.sess.run(
            [self.d_loss_real, self.d_loss_fake,
            self.g_loss, self.D, self.D_], feed_dict={self.z: batch_z,
            self.y: batch_y, self.inputs: batch_images,
          })

          TP = sum(D > 0.5)
          FN = sum(D <= 0.5)
          FP = sum(D_ > 0.5)
          recall = TP/(TP + FN)
          precision = TP/(TP + FP)
          lines = []
          line_loss = "  Iter:[%8d]  Epoch:[%2d/%2d]  Batch:[%4d/%4d]  Time: %4.4f  d_loss: %.8f  g_loss: %.8f" \
            % (counter, epoch + 1, config.epoch, idx + 1, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG)
          lines.append(line_loss)
          #print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          #  % (counter, epoch + 1, config.epoch, idx + 1, batch_idxs,
          #    time.time() - start_time, errD_fake+errD_real, errG))
          print(line_loss)
          np.set_printoptions(precision=2, suppress=True)
          #print("discriminate_for_real = {}".format(D))
          #print("discriminate_for_fake = {}".format(D_))

          line_real = "discriminate_for_real       = {:.2f}%".format(100*D.mean())
          line_fake = "discriminate_for_fake       = {:.2f}%".format(100*D_.mean())
          line_recall = "recall = {:.2f}".format(recall[0])
          line_precision = "precision = {:.2f}".format(precision[0])
          lines = lines + [line_real, line_fake, line_recall, line_precision]
          ##print("discriminate_for_real       = {:.2f}%".format(100*D.mean()))
          #print("discriminate_for_real       = {:.2f}%".format(100*D.mean()))
          ##print("discriminate_for_real_wrong = {:.2f}%".format(100*D2.mean()))
          #print("discriminate_for_fake       = {:.2f}%".format(100*D_.mean()))
          #print("recall = {:.2f}".format(recall[0]))
          #print("precision = {:.2f}".format(precision[0]))
          print(line_real)
          print(line_fake)
          print(line_recall)
          print(line_precision)
          batch_end_time = time.time()
          ETA = (batch_end_time - batch_start_time)*(int(batch_idxs)*(config.epoch - epoch - 1) + (int(batch_idxs) - idx - 1))
          hh = int(ETA//(60**2))
          mm = int(ETA%(60**2)//60)
          ss = int(ETA%60)
          line_ETA = "ETA: {} h {} m {} s".format(hh, mm, ss)
          lines.append(line_ETA)
          #print("ETA: {} h {} m {} s".format(hh, mm, ss))
          print(line_ETA)
          print()
          with open(log_file_path, "a") as handle:
              handle.write('\n'.join(lines))

        
        counter += 1
        
  def discriminator(self, image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      # We insert y in the 2nd layer in accordance with the paper "Invertible cGANs for img editing"
      y_toast = tf.expand_dims(tf.expand_dims(y, 1), 1)
      # y_toast.shape = (None, 1, 1, 6)
      y_toast = tf.tile(y_toast, [1, tf.shape(h0)[1], tf.shape(h0)[2], 1])
      # y_toast.shape = (None, 32, 32, 6)
      fat_toast = tf.concat([h0, y_toast], axis=-1)
      # fat_toast.shape = (None, 32, 32, 64+6)
      h1 = lrelu(self.d_bn1(conv2d(fat_toast, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
      #h4 = linear(tf.reshape(h3, [tf.shape(image)[0], -1]), 1, 'd_h4_lin')
      #h4 = linear(tf.reshape(h3, [tf.shape(image)[0], tf.reduce_prod(h3.get_shape()[1:])]), 1, 'd_h4_lin')
      #h4 = linear(tf.reshape(h3, [image.get_shape()[0], tf.reduce_prod(h3.get_shape()[1:])]), 1, 'd_h4_lin')
      #h4 = linear(tf.reshape(h3, [h3.get_shape()[0], tf.reduce_prod(h3.get_shape()[1:])]), 1, 'd_h4_lin')
      h4 = linear(tf.reshape(h3, [tf.shape(h3)[0], tf.reduce_prod(h3.get_shape()[1:])]), 1, 'd_h4_lin')
      # Bad ones
      #h4 = linear(tf.reshape(h3, [tf.shape(h3)[0], tf.reduce_prod(tf.shape(h3)[1:])]), 1, 'd_h4_lin')
      

      return tf.nn.sigmoid(h4), h4

  def generator(self, z, y):
    """
    input param:
      z.shape = (None, 100)
      y.shape = (None, 6)

    """
    with tf.variable_scope("generator") as scope:
      #s_h, s_w = self.output_height, self.output_width
      s_h, s_w = utils.IMG_RESOLUTION
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      # 32, 32
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      # 16, 16
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      # 8, 8
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      # 4, 4

      zy = tf.concat([z, y], axis=1)
      # zy.shape = (None, 106)

      self.z_, self.h0_w, self.h0_b = linear(
          zy, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      # carpedm20 uses a fully-connected layer to map z to
      # the dimension of first layer (4, 4, 64*8=512).
      # Unlike soumith, who achieved the same
      # using Conv2DTranspose.
      # N.B. by default, s_h16 = 4, s_w16 = 4, gf_dim = 64.

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      # Options: ReLU or Leaky ReLu
      #h0 = tf.nn.relu(self.g_bn0(self.h0))
      h0 = lrelu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [tf.shape(z)[0], s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      # Options: ReLU or Leaky ReLu
      #h1 = tf.nn.relu(self.g_bn1(self.h1))
      h1 = lrelu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [tf.shape(z)[0], s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      # Options: ReLU or Leaky ReLu
      #h2 = tf.nn.relu(self.g_bn2(h2))
      h2 = lrelu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [tf.shape(z)[0], s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      # Options: ReLU or Leaky ReLu
      #h3 = tf.nn.relu(self.g_bn3(h3))
      h3 = lrelu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [tf.shape(z)[0], s_h, s_w, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def sampler(self, z, y):
    """
    Basically, this sampler() is almost identical to generator().
    
    I am not sure if this part is redundant, but
    simply replacing sample() by generator() in the code
    won't work.

    I guess it's due to the variable scope and reuse.

    """
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      #s_h, s_w = self.output_height, self.output_width
      s_h, s_w = utils.IMG_RESOLUTION
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      zy = tf.concat([z, y], axis=1)
      h0 = tf.reshape(
          linear(zy, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      # Options: ReLU or Leaky ReLu
      #h0 = tf.nn.relu(self.g_bn0(h0, train=False))
      h0 = lrelu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [tf.shape(z)[0], s_h8, s_w8, self.gf_dim*4], name='g_h1')
      # Options: ReLU or Leaky ReLu
      #h1 = tf.nn.relu(self.g_bn1(h1, train=False))
      h1 = lrelu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [tf.shape(z)[0], s_h4, s_w4, self.gf_dim*2], name='g_h2')
      # Options: ReLU or Leaky ReLu
      #h2 = tf.nn.relu(self.g_bn2(h2, train=False))
      h2 = lrelu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [tf.shape(z)[0], s_h2, s_w2, self.gf_dim*1], name='g_h3')
      #h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      #h3 = tf.nn.relu(self.g_bn3(h3, train=False))
      h3 = lrelu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [tf.shape(z)[0], s_h, s_w, self.c_dim], name='g_h4')
      
      return tf.nn.tanh(h4)


  @property
  def model_dir(self):
    #return "{}_{}_{}_{}".format(
    #    self.dataset_name, self.batch_size,
    #    self.output_height, self.output_width)
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        *utils.IMG_RESOLUTION)

  def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
    # model_name = "DCGAN.model"
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    filename += '.b' + str(self.batch_size)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    if ckpt:
      self.saver.save(self.sess,
              os.path.join(checkpoint_dir, filename),
              global_step=step)

    if frozen:
      tf.train.write_graph(
              tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["generator_1/Tanh"]),
              checkpoint_dir,
              '{}-{:06d}_frz.pb'.format(filename, step),
              as_text=False)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints", checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      print("ckpt.model_checkpoint_path = {}".format(ckpt.model_checkpoint_path))
      print("ckpt_name = {}".format(ckpt_name))
      #print("".format())
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print("whole path to ckpt = {}".format(os.path.join(checkpoint_dir, ckpt_name)))
      counter = int(ckpt_name.split('-')[-1])
      print(" [*] Succeeded in reading {}".format(ckpt_name))
      print("counter = {}".format(counter))
      return True, counter
    else:
      print(" [*] Failed to find any checkpoint in", checkpoint_dir)
      #return False, 0
      # The 2nd return value matters little.
      return False, -1
