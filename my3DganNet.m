rng('shuffle');

if 0
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
    generator.lr = 0.0025;
    generator.momentum = 0.999;
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
    discriminator.momentum = 0.999;
    discriminator.momentum2 = 0.9;
    discriminator.BNlr = 1.0e-5;
    discriminator.weight_decay = 1e-5;

    %get batch of data;
    %real_batch_data = myBatchDataProcess(generator.batchSize);

    %initial the parameters of the generator network
    generator = myNetSetup(generator,200);
    discriminator = myNetSetup4Discriminator(discriminator,1);

else
    load 'generator_parameters.mat';
    load 'discriminator_parameters.mat';
    
    discriminator.lr = 1.0e-5;
end
%read batch data

epoch = 50;
num = 5;

param.data_path = './ModelNet40Voxel';
param.classnames = {'chair'};
real_batch_data = read_data_list(param.data_path,param.classnames,'train');

n = length(real_batch_data{1});
batch_num = n/generator.batchSize;


Loss = zeros(1,1);
Loss_D = zeros(1,1);
Loss_G = zeros(1,1);
d_Loss_fake = 1;
d_Loss_real = 0;

for total_sum = 1:4
for x=0:epoch-1
    
    shuffle_index = randperm(n);
    for j=1:num

        if abs(mean(d_Loss_fake) - mean(d_Loss_real)) >=0.6 && mean(d_Loss_real) >=0.8
            fprintf('\n d_Loss_real is %f and d_Loss_fake is %f\n  break!!!!!!!! \n',...
                mean(d_Loss_real),mean(d_Loss_fake));
            break;
        end
        
        batch_list = shuffle_index(generator.batchSize*(j-1)+1:j*generator.batchSize);
        batch = myRead_batch(real_batch_data{1}(batch_list));
%        batch_list = shuffle_index((i-1)*generator.batchSize+1:i*generator.batchSize);
%        i = i+1;
%        batch = myRead_batch(real_batch_data(batch_list));

        rand_z = rand([50,200],'single');
        
        generator = myGenerator(generator,rand_z,'forward');
        gen_output = generator.layers{6}.output;
        
%         discriminator_input = myConcatenate(batch, gen_output);
%         
%         discriminator = myDiscriminator(discriminator, discriminator_input, 'forward', 'true');
%         d_Loss_real = discriminator.layers{6}.output(1:size(batch,1),:);
%         d_Loss_real_tmp = -1.*d_Loss_real.^(-1);
%         d_Loss_fake = discriminator.layers{6}.output(size(batch,1)+1:end,:);
%         d_Loss_fake_tmp = (1-d_Loss_fake).^(-1);
%         d_mean = d_Loss_real_tmp + d_Loss_fake_tmp;
%         back_loss = mean(d_mean,1).*ones(size(discriminator.layers{6}.output),'single');
%         discriminator = myDiscriminator(discriminator, back_loss, 'backward', 'true');
        
        discriminator = myDiscriminator(discriminator,batch,'forward','true');
        d_Loss_real = discriminator.layers{6}.output;
        d_Loss_real_tmp = -1.*(d_Loss_real.^(-1));
        d_Loss = mean(d_Loss_real_tmp,1).*ones(size(d_Loss_real),'single');
        
        tmp = log(d_Loss_real);
        Loss_D(end + 1,1) = mean(tmp(:));
        
        discriminator = myDiscriminator(discriminator,d_Loss,'backward','true');           
   
        discriminator = myDiscriminator(discriminator,gen_output,'forward','true');
        d_Loss_fake = discriminator.layers{6}.output;
        d_Loss_fake_tmp = (1-d_Loss_fake).^(-1);
        d_gen_Loss = mean(d_Loss_fake_tmp,1).*ones(size(d_Loss_fake),'single');
        
        tmp = log(1-d_Loss_fake);
        Loss_G(end + 1,1) = mean(tmp(:));
        
        discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','true');
        
        tmp = log(d_Loss_real) + log(1-d_Loss_fake);
        Loss(end + 1,1) = mean(tmp(:));
                
        fprintf('Loss in D is %f ,Loss_D is %f and Loss_G is %f and Loss is %f\n',Loss(end,1),...
            Loss_D(end,1),Loss_G(end,1),abs(mean(d_Loss_fake) - mean(d_Loss_real)));
    end
    
    fprintf('\n');
    
    if 1
        rand_z = rand([50,200],'single');
        generator = myGenerator(generator,rand_z,'forward');
        gen_output_2 = generator.layers{6}.output;

        discriminator = myDiscriminator(discriminator,gen_output_2,'forward','true');
        disc_output_G = discriminator.layers{6}.output;
        tmp = log(1-disc_output_G);
        Loss_G(end + 1,1)=mean(tmp(:));

        d_gen_Loss_tmp = -1.*(1-disc_output_G).^(-1);
        d_gen_Loss = mean(d_gen_Loss_tmp,1).*ones(size(d_gen_Loss_tmp),'single');

        discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','false');
        gen_Loss = discriminator.layers{1}.dinput;

        generator = myGenerator(generator,gen_Loss,'backward');
        fprintf('finished the G network %f and back_Loss is %f \n',Loss_G(end,1),mean(d_gen_Loss(:)));
    end
end
save 'generator_parameters.mat' generator -v7.3;
save 'discriminator_parameters.mat' discriminator -v7.3;
fprintf('\nfinished the training at %s\n\n',datestr(now,13));
end