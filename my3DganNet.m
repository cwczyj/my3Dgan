
%The architecture of the generator network;
generator.layers = {
    struct('type', 'fullconnect', 'outputMaps', 512, 'kernels', 4, 'actFun', 'ReLU','stride', 1);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4,'actFun', 'sigmoid', 'stride', 2);
    struct('type', 'output');
};
generator.batchSize = 100;
generator.lr = 0.0015;
generator.momentum = 0.5;
generator.BNlr = 0.001;

epcohe = 30;

%get batch of data;
% real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator);

%read batch data
% n = length(real_batch_data);
% shuffle_index = randperm(n);
% batch_num = n/generator.batchSize;

%for i=1:batch_num
% i=randi([1,batch_num],1);    
% batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
% 
% batch = myRead_batch(real_batch_data(batch_list));
%end

%test my Generator networks;
rand_z = rand([200,100],'single');
myGenerator(generator,rand_z,'forward');