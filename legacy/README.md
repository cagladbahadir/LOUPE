# LOUPE: Learning-based Optimization of the Under-sampling PattErn 

LOUPE (Python implementation in keras/tensorflow)  simultanously optimizes the under-sampling pattern and reconstruction model for MRI. See [abstract](#abstract) and [paper](#citation) for more details.


# Training LOUPE

```python
# `true_ch` is the data (e.g. size N x 256 x 256 x 1)
lmbd = 0.980 # depends on the sparsity level

# use some custom losses
def custom_mae_pred(_, y_pred):
    regu = K.mean(K.abs(y_pred[..., 0]))
    return (1-lmbd) * regu
    
def custom_mae(y_true,y_pred):
    loss_mae = K.mean(K.abs(y_true-y_pred))
    return lmbd * loss_mae

# get model. Note we use a multi_gpu model in our training.
model = unet_leaky_two_channel(64, 3, 1)
par_model = multi_gpu_model(model, gpus=8)

# prepare some place_holder k_space (second entry for the output)
# The second loss function doesn't take this into consideration 
# as it calculates the loss on the prediction and not the ground truth
k_space = np.random.uniform(low=0.0, high=1.0, size=(len(true_ch),256,256,1)) 

# compile
par_model.compile(optimizer='Adam', loss=[custom_mae, custom_mae_pred])

# fit
history = par_model.fit(true_ch, [true_ch,k_space], validation_split=0.3, epochs=200, batch_size=32,verbose=1,callbacks=callbacks_list)
```

# Custom Layers

`LocalThresholdingLayer`: Learns the probabilistic mask p on the 2D grid

`ThresholdRandomMask`: Compares p and u and applies hard sigmoid to acquire m binary mask 

`RandomMask`: Creates a random uniform mask between 0 and 1 for the thresholding function. Corresponds to vector u in the Eq. 3 in the [paper](https://arxiv.org/abs/1901.01960). 


# Citation 

If you use the open source code, please cite:  
- Learning-based Optimization of the Under-sampling Pattern in MRI.  
Cagla D. Bahadir, Adrian V. Dalca, and Mert R. Sabuncu.  
IPMI: Information Processing in Medical Imaging (2019). [arXiv:1901.01960](https://arxiv.org/abs/1901.01960).


# Abstract
Learning-based Optimization of the Under-sampling Pattern in MRI
Acquisition of Magnetic Resonance Imaging (MRI) scans can be accelerated by under-sampling in k-space (i.e., the Fourier domain). In this paper, we consider the problem of optimizing the sub-sampling pattern in a data-driven fashion. Since the reconstruction model's performance depends on the sub-sampling pattern, we combine the two problems. For a given sparsity constraint, our method optimizes the sub-sampling pattern and reconstruction model, using an end-to-end learning strategy. Our algorithm learns from full-resolution data that are under-sampled retrospectively, yielding a sub-sampling pattern and reconstruction model that are customized to the type of images represented in the training data. The proposed method, which we call LOUPE (Learning-based Optimization of the Under-sampling PattErn), was implemented by modifying a U-Net, a widely-used convolutional neural network architecture, that we append with the forward model that encodes the under-sampling process. Our experiments with T1-weighted structural brain MRI scans show that the optimized sub-sampling pattern can yield significantly more accurate reconstructions compared to standard random uniform, variable density or equispaced under-sampling schemes.


# Trained Model Weights

Trained model weights for LOUPE and individual U-Nets for different mask configurations are available upon request due to large file sizes. Please contact Cagla Deniz Bahadir (cagladeniz94@gmail.com) for the weight files.
