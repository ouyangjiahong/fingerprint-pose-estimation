fid_train = fopen('train.txt', 'w+');
fid_test = fopen('test.txt','w+');
fid_val = fopen('val.txt','w+');
fid_train_val = fopen('trainval.txt', 'w+');

numImages = 24 * 3000;
randset = randperm(numImages);
for i = 1 : numImages
    num = randset(i);
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
    name = strcat(zero, int2str(num));
    remainder = mod(i, 4);
    if remainder == 0
        fprintf(fid_test, '%s\n', name);
    elseif remainder == 1
        fprintf(fid_test, '%s\n', name);
    elseif remainder == 2
        fprintf(fid_train, '%s\n', name);
        fprintf(fid_train_val, '%s\n', name);
    else
        fprintf(fid_val, '%s\n', name);
        fprintf(fid_train_val, '%s\n', name);
    end
end

fclose(fid_train);
fclose(fid_test);
fclose(fid_val);
fclose(fid_train_val);