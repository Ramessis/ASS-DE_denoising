function psnrvalue = rgbPSNR2(image1, image2)
% image1��image2��С���
image1 = double(image1);
image2 = double(image2);

[row, col, ~] = size(image1); % ͼ��ĳ��Ϳ�
MSE = sum(bsxfun(@minus, image1, image2).^2, 'all') / (row * col * 3); % ����MSE

MAX = double(intmax(class(image1))); % ͼ��ĻҶȼ���
B = ceil(log2(MAX + 1)); % ����һ���������ö�����λ��
psnrvalue = 20 * log10(MAX) - 10 * log10(MSE); % ����PSNR
end
