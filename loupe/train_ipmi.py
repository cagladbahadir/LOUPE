"""
    LOUPE training example (IPMI version)

    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Primary mail: cagladeniz94@gmail.com
    
    Please cite the below paper for the code:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019
    arXiv preprint arXiv:1901.01960 (2019).
"""


# imports
import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
import keras.models

# loupe
import models

###############################################################################
# parameters
###############################################################################

# TODO put them in the form of ArgumentParser()
#   see e.g. https://github.com/voxelmorph/voxelmorph/blob/master/src/train.py
gpu_id = 7  # gpu id
lmbd = 0.99 # original loss functions from LOUPE website, will affect sparsity level
models_dir = '../models/ipmi_test/' # change this to a location to save models
nb_epochs_train = 60
batch_size = 32


###############################################################################
# GPU
###############################################################################

# gpu handling
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


###############################################################################
# Data - FASHION_MNIST for demo, replace with your favorite dataset
###############################################################################

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
xdata = np.pad(x_train, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32
xdata = xdata[..., np.newaxis]/255
val_data = xdata[0:1,...]
xdata = xdata[1:,...]
vol_size = xdata.shape[1:-1]

# prepare some place_holder k_space (second entry for the output)
# The second loss function doesn't take this into consideration 
# as it calculates the loss on the prediction and not the ground truth
k_space = np.random.uniform(low=0.0, high=1.0, size=xdata.shape) 


###############################################################################
# Prepare model
###############################################################################

# train model
model = models.loupe_model(input_shape=vol_size + (1,), filt=64, kern=3, model_type='v1')

# use some custom losses
def mask_wt_l1(_, y_pred):
    return tf.reduce_mean(tf.abs(y_pred[..., 0]))

# compile
model.compile(optimizer='Adam', loss=['mae', mask_wt_l1], loss_weights=[lmbd, 1-lmbd])

# prepare save folder
if not os.path.isdir(models_dir): os.makedirs(models_dir)
filename = os.path.join(models_dir, 'model.{epoch:02d}.hdf5')

###############################################################################
# Train model
###############################################################################

# training
model.save_weights(filename.format(epoch=0))
history = model.fit(xdata, [xdata, k_space],
                    validation_split=0.3,
                    initial_epoch=1,
                    epochs=1 + nb_epochs_train,
                    batch_size=batch_size, 
                    callbacks=[ModelCheckpoint(filename)],
                    verbose=1)


###############################################################################
# View prob mask
###############################################################################

mask_filename = os.path.join(models_dir, 'mask.npy')
print('saving mask to %s' % mask_filename)
mask_model = keras.models.Model(model.inputs, model.get_layer('prob_mask').output)
prob_mask = mask_model.predict(val_data)
np.save(mask_filename, prob_mask)
