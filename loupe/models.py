"""
    LOUPE

    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Primary mail: cagladeniz94@gmail.com
    
    Please cite the below paper for the code:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019
    arXiv preprint arXiv:1901.01960 (2019).
"""

# core python
import sys

# third party
import keras.models
import numpy as np
from keras import backend as K
from keras.layers import Layer, Activation, LeakyReLU
from keras.layers import Input, AveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add

# local models
from . import layers


def loupe_model(input_shape=(256,256,1),
                filt=64,
                kern=3,
                sparsity=None,
                pmask_slope=5,
                pmask_init=None,
                sample_slope=12,
                acti=None,
                model_type='v2'):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        model_type: 'unet', 'v1', 'v2'
        sparsity: desired sparsity (only for model type 'v2')
        pmask_slope: slope of logistic parameter in probability mask
        acti=None: activation
        
    Returns:
        keras model

    UNet leaky two channel
    """
    
    
    if model_type == 'unet': # Creates a single U-Net
        inputs = Input(shape=input_shape, name='input')
        last_tensor = inputs

    elif model_type == 'loupe_mask_input':
  
        # inputs
        inputs = [Input(shape=input_shape, name='input'), Input(shape=input_shape, name='mask_input')]

        # input -> kspace via FFT
        last_tensor = layers.FFT(name='fft')(inputs[0])

        # input mask
        last_tensor_mask = inputs[1]

        # Under-sample and back to image space via IFFT
        last_tensor = layers.UnderSample(name='under_sample_kspace')([last_tensor, last_tensor_mask])
        last_tensor = layers.IFFT(name='under_sample_img')(last_tensor)

    else: # Creates LOUPE
        assert model_type in ['v1', 'v2'], 'model_type should be unet, v1 or v2'

        # inputs
        inputs = Input(shape=input_shape, name='input')

        # input -> kspace via FFT
        last_tensor = layers.FFT(name='fft')(inputs)

        # build probability mask
        prob_mask_tensor = layers.ProbMask(name='prob_mask', slope=pmask_slope, initializer=pmask_init)(last_tensor) 
        
        if model_type == 'v2':
            assert sparsity is not None, 'for this model, need desired sparsity to be specified'
            prob_mask_tensor = layers.RescaleProbMap(sparsity, name='prob_mask_scaled')(prob_mask_tensor)

        else:
            assert sparsity is None, 'for v1 model, cannot specify sparsity'
        
        # Realization of probability mask
        thresh_tensor = layers.RandomMask(name='random_mask')(prob_mask_tensor) 
        last_tensor_mask = layers.ThresholdRandomMask(slope=sample_slope, name='sampled_mask')([prob_mask_tensor, thresh_tensor]) 

        # Under-sample and back to image space via IFFT
        last_tensor = layers.UnderSample(name='under_sample_kspace')([last_tensor, last_tensor_mask])
        last_tensor = layers.IFFT(name='under_sample_img')(last_tensor)

    # hard-coded UNet
    unet_tensor = _unet_from_tensor(last_tensor, filt, kern, acti, output_nb_feats=input_shape[-1])      

    # complex absolute layer
    abs_tensor = layers.ComplexAbs(name='complex_addition')(last_tensor)

    # final output from model 
    add_tensor = Add(name='unet_output')([abs_tensor, unet_tensor])
    
    # prepare and output a model as necessary
    outputs = [add_tensor]
    if model_type == 'v1':
        outputs += [last_tensor_mask]
    
    return keras.models.Model(inputs, outputs)


def _unet_from_tensor(tensor, filt, kern, acti, output_nb_feats=1):
    """
    UNet used in LOUPE

    TODO: this is quite rigid right now and hardcoded (# layers, features, etc)
    - use for loops and such crazy things
    - use a richer library for this, perhaps neuron
    """

    # start first convolution of UNet
    conv1 = Conv2D(filt, kern, activation = acti, padding = 'same', name='conv_1_1')(tensor)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization(name='batch_norm_1_1')(conv1)
    conv1 = Conv2D(filt, kern, activation = acti, padding = 'same', name='conv_1_2')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization(name='batch_norm_1_2')(conv1)
    
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same', name='conv_2_1')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization(name='batch_norm_2_1')(conv2)
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same', name='conv_2_2')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization(name='batch_norm_2_2')(conv2)
    
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same', name='conv_3_1')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization(name='batch_norm_3_1')(conv3)
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same', name='conv_3_2')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization(name='batch_norm_3_2')(conv3)
    
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same', name='conv_4_1')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(name='batch_norm_4_1')(conv4)
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same', name='conv_4_2')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(name='batch_norm_4_2')(conv4)
    
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same', name='conv_5_1')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization(name='batch_norm_5_1')(conv5)
    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same', name='conv_5_2')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization(name='batch_norm_5_2')(conv5)

    sub1 = UpSampling2D(size=(2, 2))(conv5)
    concat1 = Concatenate(axis=-1)([conv4,sub1])
    
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same', name='conv_6_1')(concat1)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization(name='batch_norm_6_1')(conv6)
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same', name='conv_6_2')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization(name='batch_norm_6_2')(conv6)

    sub2 = UpSampling2D(size=(2, 2))(conv6)
    concat2 = Concatenate(axis=-1)([conv3,sub2])
    
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same', name='conv_7_1')(concat2)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization(name='batch_norm_7_1')(conv7)
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same', name='conv_7_2')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization(name='batch_norm_7_2')(conv7)

    sub3 = UpSampling2D(size=(2, 2))(conv7)
    concat3 = Concatenate(axis=-1)([conv2,sub3])
    
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same', name='conv_8_1')(concat3)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization(name='batch_norm_8_1')(conv8)
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same', name='conv_8_2')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization(name='batch_norm_8_2')(conv8)

    sub4 = UpSampling2D(size=(2, 2))(conv8)
    concat4 = Concatenate(axis=-1)([conv1,sub4])
    
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same', name='conv_9_1')(concat4)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization(name='batch_norm_9_1')(conv9)
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same', name='conv_9_2')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization(name='batch_norm_9_2')(conv9)
    conv9 = Conv2D(output_nb_feats, 1, padding = 'same', name='conv_9_3')(conv9)
    
    return conv9
