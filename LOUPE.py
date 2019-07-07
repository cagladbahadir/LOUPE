"""
    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Please cite the below paper for the code:
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. "Learning-based Optimization of the Under-sampling Pattern in MRI." arXiv preprint arXiv:1901.01960 (2019).
    Primary mail: cagladeniz94@gmail.com
"""

# third party
import sys
from keras import layers
import numpy as np
from keras import backend as K
from keras.legacy import interfaces
import keras
from keras.layers import Layer, Activation, Subtract
import tensorflow as tf
from keras.initializers import RandomUniform,Identity,RandomNormal
from keras.activations import softmax
from keras import activations
from keras.models import Model
from keras.constraints import maxnorm
from keras.layers import Input, Dense,Lambda, Conv3D, MaxPooling3D,UpSampling3D, Conv3DTranspose, Dropout, AveragePooling2D
from keras.layers import Reshape, Conv2D,MaxPooling2D,UpSampling2D,LocallyConnected2D
from keras.layers.merge import Concatenate, Add
from keras import losses, optimizers
from keras.initializers import RandomNormal, Identity
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils.training_utils import multi_gpu_model

# local
#from keras.utils import transform, integrate_vec, affine_to_shift


class LocalThresholdingLayer(Layer):
    """ 
    Local thresholding layer: modified from Local Linear Layer code in 
    tensorflow/keras utilities for the neuron project by
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    """
    
    def __init__(self, my_initializer=RandomUniform(minval=-15.0, maxval=15.0, seed=None),activation='softmax', **kwargs):
    
        self.initializer = my_initializer
        super(LocalThresholdingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        lst = list(input_shape)
        lst[3] = 1
        input_shape_h = tuple(lst)
        
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=input_shape_h[1:],
                                    
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(LocalThresholdingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self,x):
        k=0.25 # Slope of sigmoid
    
        output =0*x[:,:,:,0:1]+self.mult
        

        return (1/(1+K.exp(-1*k*(output))))
    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[3] = 1
        input_shape_n = tuple(lst)
   
        return input_shape_n
    
    
class ThresholdRandomMask(Layer):
    """ 
    Local thresholding layer: modified from Local Linear Layer code in 
    tensorflow/keras utilities for the neuron project by
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    """

    def __init__(self, my_initializer=RandomUniform(minval=0.0, maxval=1.0, seed=None), **kwargs):
        self.initializer = my_initializer
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self,x):
        k=12
        input_shape = tf.shape(x)
        thresh = x[:,:,:,1:2]
        inputs = x[:,:,:,0:1]
        return (1/(1+K.exp(-1*k*(inputs-thresh))))
    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[3] = 1
        input_shape_n = tuple(lst)
        return input_shape_n
    
class RandomMask(Layer):
    """ 
    Local thresholding layer: modified from Local Linear Layer code in 
    tensorflow/keras utilities for the neuron project by
    Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
    """

    def __init__(self, my_initializer=RandomUniform(minval=0.0, maxval=1.0, seed=None), **kwargs):
        self.initializer = my_initializer
        super(RandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomMask, self).build(input_shape)

    def call(self,x):
        input_shape = tf.shape(x)
        threshs = K.random_uniform(input_shape, minval=0.0,maxval=1.0, dtype='float32')

        return (0*x)+threshs
    def compute_output_shape(self, input_shape):
        
        return input_shape



from keras.layers import LeakyReLU



def unet_leaky_two_channel(filt=64,kern=3,model_type=0):
    acti=None
    if model_type==0: # Creates a single U-Net
        inputShape=(256,256,2)
        inputs =Input(shape=inputShape,)
        conv1 = Conv2D(filt, kern, activation = acti, padding = 'same')(inputs)
    else:
        inputShape=(256,256,1) # Creates LOUPE
        inputs =Input(shape=inputShape,)

        last_tensor= Lambda(Lambda_fft)(inputs) # FFT operation
        last_tensor_hold = LocalThresholdingLayer()(last_tensor) # Creates probability mask
        last_tensor_hold_thresh = RandomMask()(last_tensor_hold) # Creates thresholds
        last_tensor_mask = Concatenate(axis=-1)([last_tensor_hold,last_tensor_hold_thresh])
        last_tensor_mask = ThresholdRandomMask()(last_tensor_mask) # Creates realization of probability mask
        last_tensor = Concatenate(axis=-1)([last_tensor,last_tensor_mask])
        last_tensor = Lambda(Lambda_mult)(last_tensor) # Under-sampling operation
        last_tensor = Lambda(Lambda_ifft)(last_tensor) # IFFT operation
        conv1 = Conv2D(filt, kern, activation = acti, padding = 'same')(last_tensor)
        
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filt, kern, activation = acti, padding = 'same')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = BatchNormalization()(conv1)
    
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filt*2, kern, activation = acti, padding = 'same')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    
    
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filt*4, kern, activation = acti, padding = 'same')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filt*8, kern, activation = acti, padding = 'same')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filt*16, kern, activation = acti, padding = 'same')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)

    sub1 = UpSampling2D(size=(2, 2))(conv5)
    concat1 = Concatenate(axis=-1)([conv4,sub1])
    
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same')(concat1)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(filt*8, kern, activation = acti, padding = 'same')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)

    sub2 = UpSampling2D(size=(2, 2))(conv6)
    concat2 = Concatenate(axis=-1)([conv3,sub2])
    
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same')(concat2)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(filt*4, kern, activation = acti, padding = 'same')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)

    sub3 = UpSampling2D(size=(2, 2))(conv7)
    concat3 = Concatenate(axis=-1)([conv2,sub3])
    
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same')(concat3)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(filt*2, kern, activation = acti, padding = 'same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)

    sub4 = UpSampling2D(size=(2, 2))(conv8)
    concat4 = Concatenate(axis=-1)([conv1,sub4])
    
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same')(concat4)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(filt, kern, activation = acti, padding = 'same')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(1, 1, padding = 'same')(conv9)
    
    if model_type==0:
        abs_layer = Lambda(Abs_layer)(inputs)
    else:
        abs_layer = Lambda(Abs_layer)(last_tensor)
        
    add_1 = Add()([abs_layer,conv9])
    if model_type==0:
        model = Model(input = inputs, output = add_1)
    else:
        model = Model(input = inputs, output = [add_1,last_tensor_hold,last_tensor_mask])
    

    return model


def Abs_layer(inputs):
    two_channel = tf.complex(inputs[:,:,:,0],inputs[:,:,:,1])
    shape =tf.shape(tf.expand_dims(two_channel, -1))
    two_channel = tf.reshape(two_channel,shape)
    two_channel = tf.abs(two_channel)
    two_channel = tf.cast(two_channel, tf.float32)
    return two_channel

def Lambda_mult(inputs): # Under-sampling by multiplication of k-space with the mask
    k_space_r = tf.multiply(inputs[:,:,:,0],inputs[:,:,:,2])
    k_space_i = tf.multiply(inputs[:,:,:,1],inputs[:,:,:,2])
    shape = tf.shape(k_space_r)
    shape = tf.shape(tf.expand_dims(k_space_r, -1))
    k_space_r = tf.reshape(k_space_r,shape)
    k_space_i = tf.reshape(k_space_i,shape)
    k_space = tf.concat([k_space_r,k_space_i],axis = -1)
    k_space = tf.cast(k_space, tf.float32)
    return k_space

def Lambda_fft(input_im): # FFT layer
    input_im = input_im[:,:,:,0]
    input_im = tf.cast(input_im, tf.complex64)
    k_space = tf.fft2d(input_im)
    shape = tf.shape(k_space)
    shape =tf.shape(tf.expand_dims(k_space, -1))
    k_space = tf.reshape(k_space,shape)
    k_space_r = tf.real(k_space)
    k_space_i = tf.imag(k_space)
    k_space = tf.concat([k_space_r,k_space_i],axis = 3)
    k_space = tf.cast(k_space, tf.float32)
    return k_space

def Lambda_ifft(k_space): # IFFT layer
    k_space = tf.complex(k_space[:,:,:,0],k_space[:,:,:,1])
    output_im = tf.ifft2d(k_space)
    shape = tf.shape(output_im)
    shape =tf.shape(tf.expand_dims(output_im, -1))
    output_im = tf.reshape(output_im,shape)
    output_r = tf.real(output_im)
    output_i = tf.imag(output_im)
    output_im = tf.concat([output_r,output_i],axis = 3)
    output_im = tf.cast(output_im, tf.float32)
    return output_im







 