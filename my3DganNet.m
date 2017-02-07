rng('shuffle');

%The architecture of the generator network;
generator.layers = {
    struct('type', 'fullconnect', 'outputMaps', 512, 'kernels', 4, 'actFun', 'ReLU','stride', 1,'padding',0);
    struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
    struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4,'actFun', 'sigmoid', 'stride', 2,'padding',1);
    struct('type', 'output');
};
generator.batchSize = 10;
generator.lr = 0.0025;
generator.momentum = 0.5;
generator.BNlr = 0.001;
generator.weight_decay = 1e-5;

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
discriminator.batchSize = 10;
discriminator.lr = 1e-5;
discriminator.momentum = 0.9;
discriminator.BNlr = 0.001;
discriminator.weight_decay = 1e-5;

epoch = 5;

%get batch of data;
real_batch_data = myBatchDataProcess(generator.batchSize);

%initial the parameters of the generator network
generator = myNetSetup(generator,200);
discriminator = myNetSetup(discriminator,1);

%read batch data
n = length(real_batch_data);
shuffle_index = randperm(n);
batch_num = n/generator.batchSize;


for x=1:epoch
    i=1;
    
    for j = 1:5
        
        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
        i = i+1;
        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([200,10],'single');
        
        generator = myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.input{1};
        
        discriminator = myDiscriminator(discriminator,batch,'forward','true');
        disc_output_real = discriminator.layers{6}.input{1};
        discriminator = myDiscriminator(discriminator,gen_output,'forward','true');
        disc_output_G = discriminator.layers{6}.input{1};
        
        disc_Loss = log10(disc_output_real) + log10(1-disc_output_G);
        disc_Loss = mean(disc_Loss(:));
        disc_Loss = disc_Loss*ones(10,1);
        
        discriminator = myDiscriminator(discriminator,disc_Loss','backward','true');
    end
    
    rand_z = rand([200,10],'single');
    generator = myGenerator(generator,rand_z,'forward');
    gen_output_2 = generator.layers{6}.input{1};
    
    discriminator = myDiscriminator(discriminator,gen_output_2,'forward','true');
    
    disc_output_G = discriminator.layers{6}.input{1};
    
    disc_Loss = log10(1-disc_output_G);
    disc_Loss = mean(disc_Loss(:));
    disc_Loss = disc_Loss*ones(10,1);
    
    discriminator = myDiscriminator(discriminator,disc_Loss,'backward','false');
    gen_Loss = discriminator.layers{1}.dinput{1};
    
    generator = myGenerator(generator,gen_Loss,'backward');
end