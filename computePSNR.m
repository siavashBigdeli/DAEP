function psnr = computePSNR(gt, map, pad)

gt_u = uint8(gt(pad+1:end-pad, pad+1:end-pad,:));
map_u = uint8(map(pad+1:end-pad, pad+1:end-pad,:));
imdff = double(map_u) - double(gt_u);
rmse = sqrt(mean(imdff(:).^2));
psnr = 20*log10(255/rmse);
