
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

%get batch of data;
real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator);