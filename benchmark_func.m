function f=benchmark_func(imn,img,x)
%function to compute objective function
for j=1:size(x,1)
    
    String=cell(1,3);
    % ��ͼ��ת��Ϊ˫���ȸ�������ʽ�������й�һ������
    img = im2double(img);
    % ��ÿ����ɫͨ������С���任
    red = img(:,:,1);
    green = img(:,:,2);
    blue = img(:,:,3);
    
    % ��ÿ��ͨ����С��ϵ��������ֵ���������н���
    for i = 1:3
        th1 = x(1,1+(6*(i-1)));
        th2 = x(1,2+(6*(i-1)));
        th3 = x(1,3+(6*(i-1)));
        th4 = x(1,4+(6*(i-1)));
        wv = round(x(1,5 +(6*(i-1))));
        lv(i) = round(x(1,6 +(6*(i-1))));
        TH(i,:) = [th1 th2 th3 th4];
        THR{i,:} = repmat(TH(i,1:lv(i)),3,1);
        %С��ϵѡ��
        if wv == 1
            string = 'sym4';
        elseif wv == 2
            string = 'db4';
        elseif wv == 3
            string = 'sym6';
        elseif wv == 4
            string = 'db6';
        elseif wv == 5
            string = 'coif4';
        elseif wv == 6
            string = 'db8';
        elseif wv == 7
            string = 'sym10';
        elseif wv == 8
            string = 'coif2';
        elseif wv == 9
            string = 'sym8';
        end
        String{i} = string;
    end
    
    %�ֱ��������ɫͨ�����н��봦��
    red_denoised = wdencmp('lvd',red,String{1},lv(1),THR{1},'h');
    green_denoised = wdencmp('lvd',green,String{2},lv(2),THR{2},'h');
    blue_denoised = wdencmp('lvd',blue,String{3},lv(3),THR{3},'h');
    
    % ������ͨ���Ľ����ϳɲ�ɫͼ��
    img_denoised = cat(3,red_denoised,green_denoised,blue_denoised);
    
    %     p1 = PSNR(img,dn)/100;
    p1 = rgbPSNR(img,img_denoised)/100;
%     p1 = PSNR(img,img_denoised)/100;
    %     p2 = ssim(img,dn);
    f(j) = -p1;
end
f=f';
end