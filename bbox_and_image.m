% rotate and save images , calculate and save bboxs


load('Refinforef.mat');
dir = '/Users/ouyangjiahong/Desktop/Thesis/data/';
src_dir = strcat(dir, 'dataset_3000/');
dst_dir = strcat(dir, 'dataset_3000/rotate/');
fid = fopen('bbox.txt', 'w+');
fid2 = fopen('index.txt', 'w+');

bbox_x = 350;
bbox_y = 350;

num = 0;
bbox = [];
index = {};
for i = 1 : 5
    for theta = 0 : 15 : 359
        %theta = 30;
        tmp = tits{i};
        img_src_dir = strcat(src_dir, tmp, '.bmp');
        num = num + 1;
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
        img_dst_dir = strcat(dst_dir, zero, int2str(num),'.jpg');

        %rotate and save
        img = imread(img_src_dir);
        img_rot = imrotate(img, theta, 'crop');
        imwrite(img_rot, img_dst_dir);
        
        %get the center infomation
        [row, col, channel] = size(img);
        center_x = refs(i, 1);
        center_y = refs(i, 2);
        angle = refs(i, 3);
        theta0 = atand(center_x / center_y);
        r0 = sqrt(center_x * center_x + center_y * center_y);

        % calculate the new center point
        [row_new_crop, col_new_crop, channel] = size(img_rot);
        if theta >= 0 && theta < 90 
            row_new = col * sind(theta) + row * cosd(theta);
            col_new = row * sind(theta) + col * cosd(theta);
            center_x_new = r0 * sind(theta + theta0);
            center_y_new = col * sind(theta) + r0 * cosd(theta + theta0);
        elseif theta >= 90 && theta < 180
            row_new = row * sind(theta - 90) + col * cosd(theta - 90);
            col_new = row * cosd(theta - 90) + col * sind(theta - 90);
            center_x_new = col * sind(theta - 90) + r0 * cosd(theta0 + theta - 90);
            center_y_new = row_new - r0 * sind(theta0 + theta - 90);
        elseif theta >= 180 && theta < 270
            row_new = row * cosd(theta - 180) + col * sind(theta - 180);
            col_new = col * cosd(theta - 180) + row * sind(theta - 180);
            center_x_new = col_new - r0 * sind(theta0 + theta - 180);
            center_y_new = row * cosd(theta - 180) - r0 * cosd(theta0 + theta -180);
        else
            row_new = row * cosd(360 - theta) + col * sind(360 - theta);
            col_new = col * cosd(360 - theta) + row * sind(360 - theta);
            center_x_new = row * sind(360 - theta) - r0 * sind(360 - theta - theta0);
            center_y_new = r0 * cosd(360 -theta - theta0);
        end
        center_x_new = floor(center_x_new - (col_new - col_new_crop) / 2);
        center_y_new = floor(center_y_new - (row_new - row_new_crop) / 2);

        %calculate the rectangle
        start_x = max(center_x_new - bbox_x / 2, 1);
        start_y = max(center_y_new - bbox_y / 2, 1);
        end_x = min(center_x_new + bbox_x / 2, col_new_crop);
        end_y = min(center_y_new + bbox_y / 2, row_new_crop);
        
        %save image
        label = 'fingerprint';
        name = strcat(zero, int2str(num), '.jpg');
        fprintf(fid,'%s %s %d %d %d %d\n', name, label, start_x, start_y, ...
                end_x, end_y);
        fprintf(fid2, '%s %s %d\n', tmp, name, theta);
    end
end
