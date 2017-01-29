rng('shuffle');

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
generator.lr = 0.0025;
generator.momentum = 0.5;
generator.BNlr = 0.001;

%The architecture of the discriminator network;
discriminator.layers = {
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 512, 'kernels', 4,'actFun', 'LReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4, 'actFun', 'sigmoid','stride', 1,'padding',0);
    struct('type', 'output');
};
discriminator.LeakyReLU = 0.2;
discriminator.batchSize = 100;
discriminator.lr = 1e-5;
discriminator.momentum = 0.9;
discriminator.BNlr = 0.001;

epoch = 15;

%get batch of data;
real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator);
discriminator = myNetSetup(discriminator);

%read batch data
n = length(real_batch_data);
shuffle_index = randperm(n);
batch_num = n/generator.batchSize;


for x=1:epoch
    i=1;
    
    for j = 1:10
        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
        i = i+1;
        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([200,100],'single');
        
        myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.input{1};
        
        myDiscriminator(discriminator,batch,'forward','true');
        disc_output_real = discriminator.layers{6}.input{1};
        myDiscriminator(discriminator,gen_output,'forward','true');
        disc_output_G = discriminator.layers{6}.input{1};
        
        disc_Loss = log10(disc_output_real) + log10(1-disc_output_G);
        disc_Loss = mean(disc_Loss(:));
        disc_Loss = disc_Loss*ones(100,1);
        
        myDiscriminator(discriminator,disc_Loss,'backward','true');
    end
    
    rand_z = rand([200,100],'single');
    myGenerator(generator,rand_z,'forward');
    gen_output_2 = generator.layers{6}.input{1};
    
    myDiscriminator(discriminator,gen_output_2,'forward','true');
    
    disc_output_G = discriminator.layers{6}.input{1};
    
    disc_Loss = log10(1-disc_output_G);
    disc_Loss = mean(disc_Loss(:));
    disc_Loss = disc_Loss*ones(100,1);
    
    myDiscriminator(discriminator,disc_Loss,'backward','false');
    gen_Loss = discriminator.layers{1}.dinput{1};
    
    myGenerator(generator,gen_Loss,'backward');
end