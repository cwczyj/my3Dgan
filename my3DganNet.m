rng('shuffle');

%The architecture of the generator network;
generator.layers = {
    struct('type', 'fullconnect', 'outputMaps', 512, 'kernels', 4, 'actFun', 'ReLU','stride', 1,'padding',0);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4,'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'output');
};
generator.batchSize = 5;
generator.lr = 0.01;
generator.momentum = 0.5;
generator.BNlr = 0.001;
generator.weight_decay = 1e-5;

%The architecture of the discriminator network;
discriminator.layers = {
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 512, 'kernels', 4,'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4, 'actFun', 'LReLU','stride', 1,'padding',0);
    struct('type', 'output');
};
discriminator.LeakyReLU = 0.2;
discriminator.batchSize = 5;
discriminator.lr = 0.01;
discriminator.momentum = 0.9;
discriminator.BNlr = 0.001;
discriminator.weight_decay = 1e-5;

epoch = 5;
num = 5;

%get batch of data;
%real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator,200);
discriminator = myNetSetup(discriminator,1);

%read batch data


param.data_path = './ModelNet40Voxel';
param.classnames = {'airplane'};
real_batch_data = read_data_list(param.data_path,param.classnames,'train');

n = length(real_batch_data{1});
shuffle_index = randperm(n);
batch_num = n/generator.batchSize;

batch_list = shuffle_index(generator.batchSize+1:2*generator.batchSize);
batch = myRead_batch(real_batch_data{1}(batch_list));

Loss = zeros(num*epoch);

for x=1:epoch
%    i=1;
    
    for j = 1:num
        
%        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
%        i = i+1;
%        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([200,5],'single');
        
        generator = myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.output{1};
        
        discriminator = myDiscriminator(discriminator,batch,'forward','true');
        disc_output_real = discriminator.layers{6}.output{1};
        d_Loss = -1.*disc_output_real.^(-1);
        tmp=['d_Loss',num2str(x),'+',num2str(j),'.mat'];
        save(tmp,'d_Loss');
        discriminator = myDiscriminator(discriminator,d_Loss,'backward','true');
        
        discriminator = myDiscriminator(discriminator,gen_output,'forward','true');
        disc_output_G = discriminator.layers{6}.output{1};
        d_gen_Loss = (1-disc_output_G).^(-1);
        tmp=['d_gen_Loss',num2str(x),'+',num2str(j),'.mat'];
        save(tmp,'d_gen_Loss');
        discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','true');
        
        tmp = log(disc_output_real) + log(1.0-disc_output_G);
        Loss(x+j) = mean(tmp(:));
   end
    
    rand_z = rand([200,5],'single');
    generator = myGenerator(generator,rand_z,'forward');
    gen_output_2 = generator.layers{6}.input{1};
    
    discriminator = myDiscriminator(discriminator,gen_output_2,'forward','true');
    
    disc_output_G = discriminator.layers{6}.input{1};
    
    d_gen_Loss = -1.*disc_output_G.^(-1);
    
    discriminator = myDiscriminator(discriminator,disc_Loss,'backward','false');
    gen_Loss = discriminator.layers{1}.dinput{1};
    
    generator = myGenerator(generator,gen_Loss,'backward');
end