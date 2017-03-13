% select the labeled images from the large dataset NIST14
load('Refinforef.mat');
dir = '/Users/ouyangjiahong/Desktop/Thesis/data/';
src_dir = strcat(dir, 'NIST14/image/');
dst_dir = strcat(dir, 'dataset_3000/');
label_dir = strcat(dst_dir, 'label.txt');
fid = fopen(label_dir,'w+');
for i = 1 : 3000
    tmp = tits{i};
    img_src_dir = strcat(src_dir, tmp, '.bmp');
    img_dst_dir = strcat(dst_dir, tmp, '.bmp');
    fprintf(fid, strcat(tmp, '.bmp', '\n'));
    %disp(img_src_dir);
    copyfile(img_src_dir, img_dst_dir);
end
fclose(fid);
