## Image Restoration using Autoencoding Priors

### [Image Restoration using Autoencoding Priors](http://www.cgg.unibe.ch/publications/image-restoration-using-autoencoding-priors)

### Abstract:
We propose to leverage denoising autoencoder networks as priors to address image restoration problems. We build on the key observation that the output of an optimal denoising autoencoder is a local mean of the true data density, and the autoencoder error (the difference between the output and input of the trained autoencoder) is a mean shift vector. We use the magnitude of this mean shift vector, that is, the distance to the local mean, as the negative log likelihood of our natural image prior. For image restoration, we maximize the likelihood using gradient descent by backpropagating the autoencoder error. A key advantage of our approach is that we do not need to train separate networks for different image restoration tasks, such as non-blind deconvolution with different kernels, or super-resolution at different magnification factors. We demonstrate state of the art results for non-blind deconvolution and super-resolution using the same autoencoding prior.


![alt text](http://www.cgg.unibe.ch//publications/image-restoration-using-autoencoding-priors/copy_of_dae.jpg)

See [manuscript](https://arxiv.org/abs/1703.09964) for details of the method.


This code runs in Matlab and you need to install [MatCaffe](http://caffe.berkeleyvision.org).
### Contents:

[demo.m](https://github.com/siavashBigdeli/DAEP/blob/master/demo.m): Includes an example for non-blind image deblurring and an example for single image supper resolution.

[DAEPDeblur.m](https://github.com/siavashBigdeli/DAEP/blob/master/DAEPDeblur.m): Implements MAP function for non-blind image deblurring. Use Matlab's help function to learn about the input and output arguments.

[DAEPSR.m](https://github.com/siavashBigdeli/DAEP/blob/master/DAEPSR.m): Implements MAP function for single image super resolution. Use Matlab's help function to learn about the input and output arguments.

[loadNet.m](https://github.com/siavashBigdeli/DAEP/blob/master/loadNet.m): Loads the Caffe 'net' object with our trained DAE.

[computePSNR.m](https://github.com/siavashBigdeli/DAEP/blob/master/computePSNR.m): Computes peak signal-to-noise ratio.

[model](https://github.com/siavashBigdeli/DAEP/tree/master/model): Includes our DAE model and learned parameters.

