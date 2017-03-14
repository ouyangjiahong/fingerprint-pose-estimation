% downsample and save

load('Refinforef.mat');
dir = '/Users/ouyangjiahong/Desktop/Thesis/data/';
src_dir = strcat(dir, 'dataset_3000/');
dst_dir = strcat(dir, 'dataset_3000_downsample/');

sample_ratio = 3;
for i = 1 : 5
    tmp = tits{i};
    img_src_dir = strcat(src_dir, tmp, '.bmp');
    img_dst_dir = strcat(dst_dir, tmp, '.bmp');
    img = imread(img_src_dir);
    img_dst = downsample(img, sample_ratio);
    img_dst = transpose(img_dst);
    img_dst = downsample(img_dst, sample_ratio);
    img_dst = transpose(img_dst);
    imshow(img_dst);
    imwrite(img_dst, img_dst_dir);
end

refs(:, 1) = floor(refs(:, 1)/sample_ratio);
refs(:, 2) = floor(refs(:, 2)/sample_ratio);
save('Refinforef_downsample.mat', 'refs', 'tits');

