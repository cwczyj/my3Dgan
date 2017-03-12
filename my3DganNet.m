rng('shuffle');

if 1
%The architecture of the generator network;
generator.layers = {
    struct('type', 'fullconnect', 'outputMaps', 512, 'kernels', 4, 'actFun', 'ReLU','stride', 1,'padding',0);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4,'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'output','padding',1);
};
generator.batchSize = 50;
generator.lr = 5.0e-5;
generator.momentum = 0.9;
generator.BNlr = 5e-5;
generator.weight_decay = 1e-5;

%The architecture of the discriminator network;
discriminator.layers = {
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 512, 'kernels', 4,'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4, 'actFun', 'LReLU','stride', 1,'padding',0);
    struct('type', 'output','padding',0);
};
discriminator.LeakyReLU = 0.2;
discriminator.batchSize = 50;
discriminator.lr = 5.0e-5;
discriminator.momentum = 0.9;
discriminator.BNlr = 5e-5;
discriminator.weight_decay = 1e-5;

end

epoch = 50;
num = 10;

%get batch of data;
%real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator,200);
discriminator = myNetSetup4Discriminator(discriminator,1);

%read batch data


param.data_path = './ModelNet40Voxel';
param.classnames = {'chair'};
real_batch_data = read_data_list(param.data_path,param.classnames,'train');

n = length(real_batch_data{1});
batch_num = n/generator.batchSize;


Loss = zeros(num*epoch+1,1);
GLoss = zeros(epoch,1);

for x=0:epoch-1
%    i=1;
    j=1;

    for j=1:num
        shuffle_index = randperm(n);
        batch_list = shuffle_index(generator.batchSize+1:2*generator.batchSize);
        batch = myRead_batch(real_batch_data{1}(batch_list));
%        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
%        i = i+1;
%        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([50,200],'single');
        
        generator = myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.output;
        
        discriminator = myDiscriminator(discriminator,batch,'forward','true');
        disc_output_real = discriminator.layers{6}.output;
        d_Loss = -1.*disc_output_real.^(-1);
        discriminator = myDiscriminator(discriminator,d_Loss,'backward','true');
        
        discriminator = myDiscriminator(discriminator,gen_output,'forward','true');
        disc_output_G = discriminator.layers{6}.output;
        d_gen_Loss = (1-disc_output_G).^(-1);
        discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','true');
        
        tmp = log(disc_output_real) + log(1.0-disc_output_G);
        
        Loss(x*num+j) = mean(tmp(:));
        fprintf('Loss in D is %f\n',Loss(x*num+j));
    end
    
    fprintf('\n');
    
    rand_z = rand([50,200],'single');
    generator = myGenerator(generator,rand_z,'forward');
    gen_output_2 = generator.layers{6}.output;
    
    discriminator = myDiscriminator(discriminator,gen_output_2,'forward','true');
    disc_output_G = discriminator.layers{6}.output;
    tmp = log(1-disc_output_G);
    GLoss(x+1)=mean(tmp(:));
    fprintf('GLoss in G is %f\n',GLoss(x+1));
    
    d_gen_Loss = -1.*(1-disc_output_G).^(-1);
    
    discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','false');
    gen_Loss = discriminator.layers{1}.dinput;
    
    generator = myGenerator(generator,gen_Loss,'backward');
    fprintf('finished the G network %f\n',GLoss(x+1));
end