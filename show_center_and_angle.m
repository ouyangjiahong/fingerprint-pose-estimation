% draw the center and the orientation on the images
load('Refinforef.mat');
dir = '/Users/ouyangjiahong/Desktop/Thesis/data/';
src_dir = strcat(dir, 'dataset_3000/');
dst_dir = strcat(dir, 'dataset_3000/');

for i = 11 : 13
    tmp = tits{i};
    img_src_dir = strcat(src_dir, tmp, '.bmp');
    img_dst_dir = strcat(dst_dir, tmp, '_label.bmp');
    img = imread(img_src_dir);
    center_x = refs(i, 1);
    center_y = refs(i, 2);
    angle = refs(i, 3);
    img_new = draw(img, center_x, center_y, angle);
    %imshow(img_new);
    %disp(angle);
    %imwrite(img_new, img_dst_dir);
end
