function psnrvalue = rgbPSNR(image1,image2)
% image1��image2��С���
row=size(image1,1); % ͼ��ĳ�
col=size(image1,2); % ͼ��Ŀ�
% ע�ⲻ���������д���,�ó�������PSNRֵ���ȼ���ƫ��
image1=double(image1);
image2=double(image2);

MSE_R=double(zeros(row,col));
MSE_G=double(zeros(row,col));
MSE_B=double(zeros(row,col));
image1_R=image1(:,:,1);  % Rͨ��
image1_G=image1(:,:,2);  % Gͨ��
image1_B=image1(:,:,3);  % Bͨ��
image2_R=image2(:,:,1);
image2_G=image2(:,:,2);
image2_B=image2(:,:,3);
% ����RGBͼ������ͨ��ÿ��ͨ����MSEֵ����ƽ��ֵ
for i=1:row
    for j=1:col
        MSE_R(i,j)=(image1_R(i,j)-image2_R(i,j))^2;
        MSE_G(i,j)=(image1_G(i,j)-image2_G(i,j))^2;
        MSE_B(i,j)=(image1_B(i,j)-image2_B(i,j))^2;
    end
end
MSE_RGB=sum(MSE_R(:))+sum(MSE_G(:))+sum(MSE_B(:)); % ��RGB����ͨ�������MSEֵ���,ע��(:)���÷�
MSE=MSE_RGB/(row*col);
B=8;         % ����һ���������ö�����λ��
MAX=2^B-1;   % ͼ��ĻҶȼ���        
psnrvalue=20*log10(MAX/sqrt(MSE)); % ����ͼ��ķ�ֵ�����                     
end
