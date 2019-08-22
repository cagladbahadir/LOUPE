"""
    Layers for LOUPE
    
    For more details, please read:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019. https://arxiv.org/abs/1901.01960.
"""


# third party
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from keras.initializers import RandomUniform, RandomNormal




class RescaleProbMap(Layer):
    """
    Rescale Probability Map

    given a prob map x, rescales it so that it obtains the desired sparsity
    
    if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    if mean(x) < sparsity, one can basically do the same thing by rescaling 
                            (1-x) appropriately, then taking 1 minus the result.
    """

    def __init__(self, sparsity, **kwargs):
        self.sparsity = sparsity
        super(RescaleProbMap, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RescaleProbMap, self).build(input_shape)

    def call(self, x):
        xbar = K.mean(x)
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        
        # compute adjucement
        le = tf.cast(tf.less_equal(r, 1), tf.float32)   
        return  le * x * r + (1-le) * (1 - (1 - x) * beta)

    def compute_output_shape(self, input_shape):
        return input_shape


class ProbMask(Layer):
    """ 
    Probability mask layer
    Contains a layer of weights, that is then passed through a sigmoid.

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """
    
    def __init__(self, slope=10,
                 initializer=None,
                 **kwargs):
        """
        note that in v1 the initial initializer was uniform in [-A, +A] where A is some scalar.
        e.g. was RandomUniform(minval=-2.0, maxval=2.0, seed=None),
        But this is uniform *in the logit space* (since we take sigmoid of this), so probabilities
        were concentrated a lot in the edges, which led to very slow convergence, I think.

        IN v2, the default initializer is a logit of the uniform [0, 1] distribution,
        which fixes this issue
        """

        if initializer == None:
            self.initializer = self._logit_slope_random_uniform
        else:
            self.initializer = initializer
        self.slope = slope
        super(ProbMask, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        takes as input the input data, which is [N x ... x 2] 
        """
        # Create a trainable weight variable for this layer.
        lst = list(input_shape)
        lst[-1] = 1
        input_shape_h = tuple(lst)
        
        self.mult = self.add_weight(name='logit_weights', 
                                      shape=input_shape_h[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(ProbMask, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self,x):
        logit_weights = 0*x[..., 0:1] + self.mult
        return tf.sigmoid(self.slope * logit_weights)

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        return tuple(lst)

    def _logit_slope_random_uniform(self, shape, dtype=None):
        eps = 1e-6
        x = K.random_uniform(shape, dtype=dtype, minval=eps, maxval=1.0-eps) # [0, 1]
        
        # logit with slope factor
        return - tf.log(1. / x - 1.) / self.slope
    
    
class ThresholdRandomMask(Layer):
    """ 
    Local thresholding layer

    Takes as input the input to be thresholded, and the threshold

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope = 12, **kwargs):
        self.slope = slope # higher slope means a more step-function-like logistic function
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, x):
        inputs = x[0]
        thresh = x[1]
        return tf.sigmoid(self.slope * (inputs-thresh))

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    

class RandomMask(Layer):
    """ 
    Create a random binary mask of the same size as the input shape
    """

    def __init__(self, **kwargs):
        super(RandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomMask, self).build(input_shape)

    def call(self,x):
        input_shape = tf.shape(x)
        threshs = K.random_uniform(input_shape, minval=0.0, maxval=1.0, dtype='float32')
        return (0*x) + threshs

    def compute_output_shape(self, input_shape):
        return input_shape


class ComplexAbs(Layer):
    """
    Complex Absolute

    Inputs: [kspace, mask]
    """

    def __init__(self, **kwargs):
        super(ComplexAbs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ComplexAbs, self).build(input_shape)

    def call(self, inputs):
        two_channel = tf.complex(inputs[..., 0], inputs[..., 1])
        shape = tf.shape(tf.expand_dims(two_channel, -1))
        two_channel = tf.reshape(two_channel, shape)
        two_channel = tf.abs(two_channel)
        two_channel = tf.cast(two_channel, tf.float32)
        return two_channel

    def compute_output_shape(self, input_shape):
        list_input_shape = list(input_shape)
        list_input_shape[-1] = 1
        return tuple(list_input_shape)


class UnderSample(Layer):
    """
    Under-sampling by multiplication of k-space with the mask

    Inputs: [kspace, mask]
    """

    def __init__(self, **kwargs):
        super(UnderSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UnderSample, self).build(input_shape)

    def call(self, inputs):
        k_space_r = tf.multiply(inputs[0][..., 0], inputs[1][..., 0])
        k_space_i = tf.multiply(inputs[0][..., 1], inputs[1][..., 0])
        shape = tf.shape(tf.expand_dims(k_space_r, -1))
        k_space_r = tf.reshape(k_space_r,shape)
        k_space_i = tf.reshape(k_space_i,shape)
        k_space = tf.concat([k_space_r, k_space_i], axis = -1)
        k_space = tf.cast(k_space, tf.float32)
        return k_space

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FFT(Layer):
    """
    ifft layer
    """

    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FFT, self).build(input_shape)

    def call(self, input_im):
        input_im = input_im[..., 0]
        input_im = tf.cast(input_im, tf.complex64)
        k_space = tf.fft2d(input_im)
        shape = tf.shape(tf.expand_dims(k_space, -1))
        k_space = tf.reshape(k_space, shape)
        k_space = tf.concat([tf.real(k_space), tf.imag(k_space)], axis=-1)
        k_space = tf.cast(k_space, tf.float32)
        return k_space

    def compute_output_shape(self, input_shape):
        list_input_shape = list(input_shape)
        list_input_shape[-1] = 1
        return tuple(list_input_shape)


class IFFT(Layer):
    """
    ifft layer
    """

    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IFFT, self).build(input_shape)

    def call(self, k_space):
        k_space = tf.complex(k_space[..., 0], k_space[..., 1])
        output_im = tf.ifft2d(k_space)

        shape = tf.shape(tf.expand_dims(output_im, -1))
        output_im = tf.reshape(output_im, shape)
        output_im = tf.concat([tf.real(output_im), tf.imag(output_im)], axis=-1)
        output_im = tf.cast(output_im, tf.float32)
        return output_im

    def compute_output_shape(self, input_shape):
        list_input_shape = list(input_shape)
        list_input_shape[-1] = 2
        return tuple(list_input_shape)
