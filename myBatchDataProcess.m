function [ output ] = myBatchDataProcess(batch_size)
%MYBATCHDATAPROCESS Summary of this function goes here
%   Preprocess the data for batch sampling operation;

%   The max number of voxels in a class is 891;
%     fileFolder = fullfile(param.data_path);
%     dirFolder = dir(fileFolder);
%     dirtmp = {dirFolder.name}';
%     
%     maxnum=0;
%     for i=3:length(dirtmp)
%         tmp.data_path=[param.data_path '/' char(dirtmp(i)) '/train'];
%         tmp.dirFolder=dir(fullfile(tmp.data_path,'*'));
%         tmp.filenum=length({tmp.dirFolder.name}');
%         fprintf('Number of class %s is %d\n',char(dirtmp(i)),tmp.filenum);
%         if tmp.filenum>maxnum
%             maxnum=tmp.filenum;
%         end
%     end
%     
%     output = maxnum;

    param.data_path = './ModelNet40Voxel';
    param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet', ...
           'airplane', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'cone', 'cup', 'curtain', 'door', ...
           'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'person', 'piano', 'plant', ...
           'radio', 'range_hood', 'sink', 'stairs', 'stool', 'tent', 'tv_stand', 'vase', 'wardrobe', 'xbox'};
    data_list = read_data_list(param.data_path,param.classnames,'train');
    output = balance_data(data_list,batch_size);
end

