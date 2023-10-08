function Wavelet()
img = (im2double((imread('D:\����\����\����Ӧ��-ң��\����ͼƬ\2.bmp'))));
imn = imnoise(img,'salt & pepper',0.1);
red = imn(:,:,1);
green = imn(:,:,2);
blue = imn(:,:,3);
    
[THR_r,SORH_r,KEEPAPP_r] = ddencmp('den','wv',red);
[THR_g,SORH_g,KEEPAPP_g] = ddencmp('den','wv',green);
[THR_b,SORH_b,KEEPAPP_b] = ddencmp('den','wv',blue);

%�ֱ��������ɫͨ�����н��봦��
red_denoised = wdencmp('gbl',red,'sym4',2,THR_r,SORH_r,KEEPAPP_r);
green_denoised = wdencmp('gbl',green,'sym4',2,THR_g,SORH_g,KEEPAPP_g);
blue_denoised = wdencmp('gbl',blue,'sym4',2,THR_b,SORH_b,KEEPAPP_b);

% ������ͨ���Ľ����ϳɲ�ɫͼ��
img_denoised = cat(3,red_denoised,green_denoised,blue_denoised);
imshow(img_denoised);
p1 = rgbPSNR(img,img_denoised)/100;
end