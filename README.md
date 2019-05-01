# LOUPE
Learning-based Optimization of the Under-sampling Pattern in MRI
Acquisition of Magnetic Resonance Imaging (MRI) scans can be accelerated by under-sampling in k-space (i.e., the Fourier domain). In this paper, we consider the problem of optimizing the sub-sampling pattern in a data-driven fashion. Since the reconstruction model's performance depends on the sub-sampling pattern, we combine the two problems. For a given sparsity constraint, our method optimizes the sub-sampling pattern and reconstruction model, using an end-to-end learning strategy. Our algorithm learns from full-resolution data that are under-sampled retrospectively, yielding a sub-sampling pattern and reconstruction model that are customized to the type of images represented in the training data. The proposed method, which we call LOUPE (Learning-based Optimization of the Under-sampling PattErn), was implemented by modifying a U-Net, a widely-used convolutional neural network architecture, that we append with the forward model that encodes the under-sampling process. Our experiments with T1-weighted structural brain MRI scans show that the optimized sub-sampling pattern can yield significantly more accurate reconstructions compared to standard random uniform, variable density or equispaced under-sampling schemes.

# CUSTOM LAYERS

LocalThresholdingLayer: Learns the probabilistic mask p on the 2D grid

ThresholdRandomMask: Compares p and u and applies hard sigmoid to acquire m binary mask 

RandomMask: Corresponds to vector u in the Eq.3 the paper https://arxiv.org/abs/1901.01960. Creates a random uniform mask between 0 and 1 for the thresholding function

# HOW TO CALL THE FUNCTION

```python
lmbd = 0.980 # depends on the sparsity level

model = unet_leaky_two_channel(64,3,1)

def custom_mse(y_true, y_pred):
    regu = K.mean(K.abs(y_pred[:,:,:,0]))
    return (1-lmbd) * regu
    
def custom_mae(y_true,y_pred):
    loss_mae = K.mean(K.abs(y_true-y_pred))
    return lmbd * loss_mae

k_space = np.random.uniform(low=0.0, high=1.0, size=(len(true_ch),256,256,1)) #second entry for the output which is a pseudo entry to be able to have 2 loss functions. The second loss function doesn't take this into consideration as it calculates the loss on the prediction and not the ground truth

par_model = multi_gpu_model(model, gpus=8)

par_model.compile(optimizer='Adam', loss=[custom_mae,custom_mse])

history = par_model.fit(true_ch, [true_ch,k_space], validation_split=0.3, epochs=200, batch_size=32,verbose=1,callbacks=callbacks_list)
```

# CITATION 

Please cite the paper for usage of the open source code in publications: Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. "Learning-based Optimization of the Under-sampling Pattern in MRI." arXiv preprint arXiv:1901.01960 (2019).

# TRAINED MODEL WEIGHTS

Trained model weights for LOUPE and individual U-Nets for different mask configurations are available upon request due to large file sizes. Please contact Cagla Deniz Bahadir (cagladeniz94@gmail.com) for the weight files.
