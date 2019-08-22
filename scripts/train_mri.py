
"""
    LOUPE training example (v2) with MR slices

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
import loupe


###############################################################################
# parameters
###############################################################################

# TODO put them in the form of ArgumentParser()
#   see e.g. https://github.com/voxelmorph/voxelmorph/blob/master/src/train.py

# setup
desired_sparsity = 0.05      # desired sparsity
pmask_slope = 5              # slope for prob mask sigmoid
sample_slope = 10            # slope after sampling via uniform mask
loss = 'mae'                 # loss

# running
gpu_id = 0                   # gpu id
nb_epochs_train = 10         # number of epochs to train for
batch_size = 16              # batch size
lr = 0.001                   # learning rate

# paths
filename_prefix = 'loupe_v2' # prefix for saving models, etc
models_dir = '../../models/' # directory to save models to
data_path = '/path/to/your/data/'  # directory of your data. See next section.


###############################################################################
# Data - 2D MRI slices
###############################################################################

# our data for this demo is stored in npz files. 
# Please change this to suit your needs
print('loading data...')
files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
xdata = np.stack([np.load(f)['vol_data'] for f in files], 0)[..., np.newaxis]
vol_size = xdata.shape[1:-1]
print('done')


###############################################################################
# GPU
###############################################################################

# gpu handling
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


###############################################################################
# Prepare model
###############################################################################

model = loupe.models.loupe_model(input_shape=vol_size + (1,),
                                      filt=64,
                                      kern=3,
                                      model_type='v2',
                                      pmask_slope=pmask_slope,
                                      sparsity=desired_sparsity,
                                      sample_slope=sample_slope)

# compile
model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=loss)

# prepare save sub-folder
local_name = '{prefix}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}'.format(
    prefix=filename_prefix,
    loss=loss,
    pmask_slope=pmask_slope,
    sample_slope=sample_slope,
    sparsity=desired_sparsity,
    lr=lr)
save_dir_loupe = os.path.join(models_dir, local_name)
if not os.path.isdir(save_dir_loupe): os.makedirs(save_dir_loupe)
filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')
print('model filenames: %s' % filename)


###############################################################################
# Train model
###############################################################################

# training
model.save_weights(filename.format(epoch=0))
history = model.fit(xdata, xdata,
                    validation_split=0.3,
                    initial_epoch=1,
                    epochs=1 + nb_epochs_train,
                    batch_size=batch_size, 
                    callbacks=[ModelCheckpoint(filename)],
                    verbose=1)


###############################################################################
# View prob mask
###############################################################################

mask_filename = os.path.join(save_dir_loupe, 'mask.npy')
print('saving mask to %s' % mask_filename)
mask_model = keras.models.Model(model.inputs, model.get_layer('prob_mask_scaled').output)
prob_mask = mask_model.predict(xdata[0:1,...])
np.save(mask_filename, prob_mask)
