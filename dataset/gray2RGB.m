% change gray images into RGB

dir = '/Users/ouyangjiahong/Desktop/Thesis/data/';
src_dir = strcat(dir, 'dataset_3000/rotate/');
dst_dir = strcat(dir, 'dataset_3000/color/');

for num = 1 : 5
    if num >= 100000
            zero = '';
    elseif num >= 10000
        zero = '0';
    elseif num >= 1000
        zero = '00';
    elseif num >= 100
        zero = '000';
    elseif num >= 10
        zero = '0000';
    else
        zero = '00000';
    end           
    name = strcat(zero, int2str(num), '.jpg');
    
    img_gray = imread(strcat(src_dir, name));
    img_rgb(:, :, 1) = img_gray;
    img_rgb(:, :, 2) = img_gray;
    img_rgb(:, :, 3) = img_gray;
    imwrite(img_rgb, strcat(dst_dir, name));
end