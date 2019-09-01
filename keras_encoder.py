import keras
from keras import layers
from keras.initializers import TruncatedNormal, RandomNormal
import numpy as np
import os
import matplotlib.pyplot as plt

def create_encoder(latent='z'):
  """
  args:
    latent, str
      Either 'z' or 'y', specifying what the encoder is for.
  
  
  returns:
    model, keras.models.Sequential
  """
  model = keras.models.Sequential()
  n_filters = 32
  for layer_i in range(4):
    if layer_i == 0:
      model.add(layers.Conv2D(filters=n_filters,
                              kernel_size=5,
                              strides=2,
                              kernel_initializer=TruncatedNormal(stddev=0.02,),
                              padding='same',
                              input_shape=(64, 64, 3),))
    else:
      model.add(layers.Conv2D(filters=n_filters * (2**layer_i),
                              kernel_size=5,
                              strides=2,
                              kernel_initializer=TruncatedNormal(stddev=0.02,),
                              padding='same',))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())

  model.add(layers.Flatten())
  optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

  if latent == 'z':
    model.add(layers.Dense(4096,
                           activation='relu',
                           kernel_initializer=RandomNormal(stddev=0.02,),
                           bias_initializer='zeros',))
    model.add(layers.Dense(100,
                           kernel_initializer=RandomNormal(stddev=0.02,),
                           bias_initializer='zeros',))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
  elif latent == 'y':
    model.add(layers.Dense(512,
                           activation='relu',
                           kernel_initializer=RandomNormal(stddev=0.02,),
                           bias_initializer='zeros',))
    model.add(layers.Dense(6,
                           activation='softmax',
                           kernel_initializer=RandomNormal(stddev=0.02,),
                           bias_initializer='zeros',))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  else:
    raise ValueError("Input arg latent has to be one of the two following strings: 'z' or 'y'.")

  return model



if __name__ == '__main__':
  # Instantiate our encoders
  Ez = create_encoder(latent='z')
  Ey = create_encoder(latent='y')

  # Load our datasets
  train_npz = os.path.join('encoder-dataset', 'train.npz')
  val_npz = os.path.join('encoder-dataset', 'val.npz')
  train_arxiv = np.load(train_npz)
  val_arxiv = np.load(val_npz)
  x_train = train_arxiv['x_train']
  y_train = train_arxiv['y_train']
  z_train = train_arxiv['z_train']
  x_val = val_arxiv['x_val']
  y_val = val_arxiv['y_val']
  z_val = val_arxiv['z_val']

  # Construct checkpoints
  #save_dir = "encoder_train_result"
  save_dir = "encoder-train-result"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  ckpt_Ez = keras.callbacks.ModelCheckpoint(os.path.join(save_dir,
                                                         "ckpt_Ez.h5"),
                                            save_best_only=True)
  ckpt_Ey = keras.callbacks.ModelCheckpoint(os.path.join(save_dir,
                                                         "ckpt_Ey.h5"),
                                            save_best_only=True)
  
  early_stopping_cb = keras.callbacks.EarlyStopping(patience=15,
                                                    restore_best_weights=True)
  # (?1) I'm not sure if we need two distinct EarlyStopping callbacks
  # for the two diff fit()


  # Start our training
  #n_epochs = 20
  n_epochs = 100
  print("Ez starts training *************************")
  print()
  history_Ez = Ez.fit(x_train,
                      z_train,
                      epochs=n_epochs,
                      batch_size=64,
                      validation_data=(x_val, z_val),
                      callbacks=[ckpt_Ez, early_stopping_cb],
                      )

  print("Ey starts training *************************")
  print()
  history_Ey = Ey.fit(x_train,
                      y_train,
                      epochs=n_epochs,
                      batch_size=64,
                      validation_data=(x_val, y_val),
                      callbacks=[ckpt_Ey, early_stopping_cb],
                      )

  loss_Ez = history_Ez.history['loss']
  val_loss_Ez = history_Ez.history['val_loss']
  #epochs_axis = range(1, n_epochs+1)
  epochs_axis = range(1, len(loss_Ez) + 1)
  plt.plot(epochs_axis, loss_Ez, 'go', label="Train Loss")
  plt.plot(epochs_axis, val_loss_Ez, 'b', label="Val Loss")
  plt.title("For Encoder E_z")
  plt.xlabel("Epochs")
  plt.ylabel("")
  plt.legend()
  plt.savefig(os.path.join(save_dir, "Ez_loss"))

  plt.clf()
  loss_Ey = history_Ey.history['loss']
  val_loss_Ey = history_Ey.history['val_loss']
  #epochs_axis = range(1, n_epochs+1)
  epochs_axis = range(1, len(loss_Ey) + 1)
  plt.plot(epochs_axis, loss_Ey, 'go', label="Train Loss")
  plt.plot(epochs_axis, val_loss_Ey, 'b', label="Val Loss")
  plt.title("For Encoder E_y")
  plt.xlabel("Epochs")
  plt.ylabel("")
  plt.legend()
  plt.savefig(os.path.join(save_dir, "Ey_loss"))



