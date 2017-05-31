
% add MatCaffe path
addpath /mnt/data/siavash/caffe/matlab;

% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;


%% Deblurring demo

% load image and kernel
load('kernels.mat');

gt = double(imread('101085.jpg'));
w = size(gt,2); w = w - mod(w, 2);
h = size(gt,1); h = h - mod(h, 2);
gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...

kernel = kernels{1};
sigma_d = 255 * .01;

degraded = convn(gt, rot90(kernel,2), 'valid');
noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;

% load network for solver
params.net = loadNet(size(gt), use_gpu);
params.gt = gt;

% run DAEP
map_deblur = DAEPDeblur(degraded, kernel, sigma_d, params);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Blurry')
subplot(133);
imshow(map_deblur/255); title('Restored')

%% Super resolution demo

up_scale = 4;
gt = double(imread('101085.jpg'));
w = size(gt,2); w = w - mod(w, up_scale*2);
h = size(gt,1); h = h - mod(h, up_scale*2);
gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...

degraded = imresize(gt, 1/up_scale, 'bicubic');
params.net = loadNet(size(gt), use_gpu);
params.gt = gt;

map_sr = DAEPSR(degraded, up_scale, params);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Low-resolution')
subplot(133);
imshow(map_sr/255); title('Supper-resolved')
