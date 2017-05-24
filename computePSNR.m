function psnr = computePSNR(im1, im2, pad)
% Implements stochastic gradient descent (SGD) for maximum-a-posteriori for image super resolution described in:
% In the paper "Image Restoration using Autoencoding Priors".
%
%
% Input:
% im1: First image in range of [0, 255].
% im1: Second image in range of [0, 255].
% pad: Scalar radious to exclude boundries from contributing to PSNR computation.
%
%
% Output: PSNR

im1_u = uint8(im1(pad+1:end-pad, pad+1:end-pad,:));
im2_u = uint8(im2(pad+1:end-pad, pad+1:end-pad,:));
imdff = double(im1_u) - double(im2_u);
rmse = sqrt(mean(imdff(:).^2));
psnr = 20*log10(255/rmse);
