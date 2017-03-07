function [ batch ] = myRead_batch( file_list )
%MYREAD_BATCH Summary of this function goes here
%   read a batch of data from data_list

data_size = 16;

batch_size = length(file_list);
batch = zeros([data_size,data_size,data_size,batch_size],'single');
for i=1:batch_size
    tmp =load(file_list(i).filename);
    batch(:,:,:,i) = tmp.Volume;
end

end

