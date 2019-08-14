# LOUPE
## Learning-based Optimization of the Under-sampling PattErn 



Python implementation in keras/tensorflow of LOUPE, which simultanously optimizes the under-sampling pattern and reconstruction model for MRI. See [abstract](#abstract) and [paper](#citation) for more details.


# Training 
Please see [train_ipmi.py](loupe/train_ipmi.py) for an example of how to train LOUPE (this uses FASHION_MNIST as an example dataset).

# Citation 

If you use the open source code, please cite:  
- Learning-based Optimization of the Under-sampling Pattern in MRI.  
Cagla D. Bahadir, Adrian V. Dalca, and Mert R. Sabuncu.  
IPMI: Information Processing in Medical Imaging (2019). [arXiv:1901.01960](https://arxiv.org/abs/1901.01960).

# Legacy Code (v1.0)
Code for the original LOUPE code was moved to the [legacy](legacy) folder.

# Abstract
Learning-based Optimization of the Under-sampling Pattern in MRI
Acquisition of Magnetic Resonance Imaging (MRI) scans can be accelerated by under-sampling in k-space (i.e., the Fourier domain). In this paper, we consider the problem of optimizing the sub-sampling pattern in a data-driven fashion. Since the reconstruction model's performance depends on the sub-sampling pattern, we combine the two problems. For a given sparsity constraint, our method optimizes the sub-sampling pattern and reconstruction model, using an end-to-end learning strategy. Our algorithm learns from full-resolution data that are under-sampled retrospectively, yielding a sub-sampling pattern and reconstruction model that are customized to the type of images represented in the training data. The proposed method, which we call LOUPE (Learning-based Optimization of the Under-sampling PattErn), was implemented by modifying a U-Net, a widely-used convolutional neural network architecture, that we append with the forward model that encodes the under-sampling process. Our experiments with T1-weighted structural brain MRI scans show that the optimized sub-sampling pattern can yield significantly more accurate reconstructions compared to standard random uniform, variable density or equispaced under-sampling schemes.


# Trained Model Weights

Trained model weights for LOUPE and individual U-Nets for different mask configurations are available upon request due to large file sizes. Please contact Cagla Deniz Bahadir (cagladeniz94@gmail.com) for the weight files.
