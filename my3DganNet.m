rng('shuffle');

if 1
    %The architecture of the generator network;
    generator.layers = {
        struct('type', 'convolution', 'outputMaps', 512, 'kernels', 4, 'actFun', 'ReLU','stride', 1,'padding',0);
        struct('type', 'convolution', 'outputMaps', 256, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
        struct('type', 'convolution', 'outputMaps', 128, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
        struct('type', 'convolution', 'outputMaps', 64, 'kernels', 4, 'actFun', 'ReLU', 'stride', 2,'padding',1);
        struct('type', 'convolution', 'outputMaps', 1, 'kernels', 4,'actFun', 'ReLU', 'stride', 2,'padding',1);
        struct('type', 'output','padding',1);
    };
    generator.batchSize = 50;
    generator.lr = 0.0025;
    generator.momentum = 0.9;
    generator.momentum2 = 0.9;
    generator.BNlr = 1.0e-5;
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
    discriminator.lr = 1.0e-5;
    discriminator.momentum = 0.9;
    discriminator.momentum2 = 0.9;
    discriminator.BNlr = 1.0e-5;
    discriminator.weight_decay = 1e-5;

    %get batch of data;
    %real_batch_data = myBatchDataProcess(generator.batchSize);

    %initial the parameters of the generator network
    generator = myNetSetup(generator,256);
    discriminator = myNetSetup4Discriminator(discriminator,1);
    
    param.data_path = './ModelNet40Voxel';
    param.classnames = {'chair'};
    real_batch_data = read_data_list(param.data_path,param.classnames,'train');
    
    d_Loss_real = 0;
    d_Loss_fake = 1;
    Loss = zeros(1,1);
    Loss_D = zeros(1,1);
    Loss_G = zeros(1,1);
else
    load('generator_parameters.mat', 'generator');
    load('discriminator_parameters.mat','discriminator');
    load('real_batch_data.mat','real_batch_data');
    load('Loss.mat','Loss');
    load('Loss_G.mat','Loss_G');
    load('Loss_D.mat','Loss_D');
    load('d_Loss_fake.mat','d_Loss_fake');
    load('d_Loss_real.mat','d_Loss_real');
end
%read batch data

epoch = 100;
num = 5;

n = length(real_batch_data{1});
batch_num = n/generator.batchSize;

for total_sum = 1:10
for x=0:epoch-1
    
    shuffle_index = randperm(n);
    
    discriminator = initialMeanBN(discriminator);
    train_times = 0;
    for j=1:num
        
        batch_list = shuffle_index(generator.batchSize*(j-1)+1:j*generator.batchSize);
        batch = myRead_batch(real_batch_data{1}(batch_list));
%        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
%        i = i+1;
%        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([50,1,1,1,256],'single');
        
        generator = myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.output;
        
        if 1
        
            discriminator_input = myConcatenate(batch, gen_output);
            discriminator = myDiscriminator(discriminator, discriminator_input, 'forward', 'true', 'train');
            d_Loss_real = discriminator.layers{6}.output(1:size(batch,1),:);
            d_Loss_real_tmp = -1.*d_Loss_real.^(-1);
            d_Loss_fake = discriminator.layers{6}.output(size(batch,1)+1:end,:);
            d_Loss_fake_tmp = (1-d_Loss_fake).^(-1);
            d_mean = d_Loss_real_tmp + d_Loss_fake_tmp;
            back_loss = zeros(size(discriminator.layers{6}.output),'single');
            back_loss(1:size(batch,1),:) = d_Loss_real_tmp;
            back_loss(size(batch,1)+1:end,:) = d_Loss_fake_tmp;
            
            train_times = train_times +1;
            if abs(mean(d_Loss_fake) - mean(d_Loss_real)) >=0.4 && mean(d_Loss_real) >=0.9
                fprintf('\n d_Loss_real is %f and d_Loss_fake is %f\n  break!!!!!!!! \n',...
                    mean(d_Loss_real),mean(d_Loss_fake));
                break;
            end
            
            discriminator = myDiscriminator(discriminator, back_loss, 'backward', 'true','train');
        
        else
        
            discriminator = myDiscriminator(discriminator,batch,'forward','train');
            d_Loss_real = discriminator.layers{6}.output;
            d_Loss_real_tmp = -1.*(d_Loss_real.^(-1));
%            d_Loss = mean(d_Loss_real_tmp,1).*ones(size(d_Loss_real),'single');
            discriminator = myDiscriminator(discriminator,d_Loss_real_tmp,'backward','true','train'); 

            discriminator = myDiscriminator(discriminator,gen_output,'forward','train');
            d_Loss_fake = discriminator.layers{6}.output;
            d_Loss_fake_tmp = (1-d_Loss_fake).^(-1);
%            d_gen_Loss = mean(d_Loss_fake_tmp,1).*ones(size(d_Loss_fake),'single');
            discriminator = myDiscriminator(discriminator,d_Loss_fake_tmp,'backward','true','train');
        
        end
        
        tmp = log(d_Loss_real);
        Loss_D(end + 1,1) = mean(tmp(:));
        tmp = log(1-d_Loss_fake);
        Loss_G(end + 1,1) = mean(tmp(:));

        tmp = log(d_Loss_real) + log(1-d_Loss_fake);
        Loss(end + 1,1) = mean(tmp(:));
                
        fprintf('Loss in D is %f ,Loss_D is %f and Loss_G is %f and Loss is %f\n',Loss(end,1),...
            Loss_D(end,1),Loss_G(end,1),abs(mean(d_Loss_fake) - mean(d_Loss_real)));
        fprintf('d_Loss_real is %f and d_Loss_fake is %f\n',...
                mean(d_Loss_real),mean(d_Loss_fake));
    end
    
    if train_times ~= 0
        discriminator = getMeanBN(discriminator,train_times);
    end
    fprintf('\n');
    
    if 1
        rand_z = rand([50,1,1,1,256],'single');
        generator = myGenerator(generator,rand_z,'forward');
        gen_output_2 = generator.layers{6}.output;

        discriminator = myDiscriminator(discriminator,gen_output_2,'forward', 'false','test');
        d_Loss_fake = discriminator.layers{6}.output;
        tmp = log(1-d_Loss_fake);
        Loss_G(end + 1,1)=mean(tmp(:));

        d_gen_Loss_tmp = -1.*(1-d_Loss_fake).^(-1);
        d_gen_Loss = d_gen_Loss_tmp;%mean(d_gen_Loss_tmp,1).*ones(size(d_Loss_fake),'single');

        discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','false','test');
        gen_Loss = discriminator.layers{1}.dinput;

        generator = myGenerator(generator,gen_Loss,'backward');
        fprintf('finished the G network %f and d_Loss_fake is %f \n',Loss_G(end,1),mean(d_Loss_fake(:)));
    end
end
save 'generator_parameters.mat' generator -v7.3;
save 'discriminator_parameters.mat' discriminator -v7.3;
save 'd_Loss_real.mat' d_Loss_real;
save 'd_Loss_fake.mat' d_Loss_fake;
save 'Loss_G.mat' Loss_G;
save 'Loss_D.mat' Loss_D;
save 'Loss.mat' Loss;
fprintf('\nfinished the training at %s\n\n',datestr(now,13));
end