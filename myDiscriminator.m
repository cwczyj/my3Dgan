function [ y ] = myDiscriminator( net, x ,forward_or_backward , update , train_or_test )
%MYDISCRIMINATOR Summary of this function goes here
%   The network of discriminator
%   update is the flag for the net to judge wether to update weights in the
%   network;
%   x is the input of the network, when this is forwardfeed x is a batch of
%   voxel data (64x64x64x100); when this is backward x is a batch of the
%   Loss (1x100) for the GAN net, L = mean(log(D(x))+log(1-D(G(z))));

global kConv_backward kConv_backward_r kConv_forward_r kConv_forward kConv_forward_c kConv_weight_r kConv_weight kConv_weight_c kConv_backward_my;

if strcmp(forward_or_backward,'forward')
    %% for Discriminator ff
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,1);
    net.layers{1}.layerSize = size(x,2);
    net.layers{1}.input = x;
    batchSizeForCompute = 100;
    
    for i=1:numel(net.layers)
        if strcmp(net.layers{i}.type,'fullconnect')
            % as if there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution') 
            %forward   
            net.layers{i+1}.layerSize = (net.layers{i}.layerSize-net.layers{i}.kernels+2*net.layers{i}.padding)/net.layers{i}.stride+1;
            
            net.layers{i}.ReLUin = zeros(batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                net.layers{i+1}.layerSize,net.layers{i}.outputMaps,'single');
            
            if size(net.layers{i}.input, 5) == 1
                kConv = kConv_forward;
            elseif size(net.layers{i}.w, 1) ~= 1
                kConv = kConv_forward_c;
            elseif size(net.layers{i}.w, 1) == 1
                kConv = kConv_forward_r;
            end
            
            tmp = zeros([batch_size,size(net.layers{i}.input,2) + 2 * net.layers{i}.padding,size(net.layers{i}.input,3) + 2 * net.layers{i}.padding,...
                size(net.layers{i}.input,4) + 2 * net.layers{i}.padding,size(net.layers{i}.input,5)],'single');
            tmp(:,1 + net.layers{i}.padding:end - net.layers{i}.padding,1 + net.layers{i}.padding:end - net.layers{i}.padding,...
                1 + net.layers{i}.padding:end - net.layers{i}.padding,:) = net.layers{i}.input;
            
            if(batch_size > batchSizeForCompute)
                iterate_num = ceil(size(net.layers{i}.input,1)/batchSizeForCompute);
                for tmp_x=1:iterate_num
                    if tmp_x == iterate_num
                        start_num = (tmp_x-1) * batchSizeForCompute + 1;
                        end_num = size(net.layers{i}.input,1);
                    else
                        start_num = (tmp_x-1) * batchSizeForCompute + 1;
                        end_num = tmp_x * batchSizeForCompute;
                    end
                    
                    tmpin = zeros(end_num-start_num,size(tmp,2),size(tmp,3),size(tmp,4),...
                        size(tmp,5),'single');         
                    for tmp_batch = start_num:end_num
                        tmpin(tmp_batch - (tmp_x-1) * batchSizeForCompute,:,:,:,:)=tmp(tmp_batch,:,:,:,:);
                    end
                    
                    tmpout = zeros(end_num-start_num,size(net.layers{i}.ReLUin,2),size(net.layers{i}.ReLUin,3),...
                        size(net.layers{i}.ReLUin,4),size(net.layers{i}.ReLUin,5),'single');
                    tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'forward');

                    for tmp_batch = start_num:end_num
                        net.layers{i}.ReLUin(tmp_batch,:,:,:,:) = tmpout(tmp_batch - (tmp_x-1) * batchSizeForCompute,:,:,:,:);
                    end
                end
            else
                net.layers{i}.ReLUin = myGPUConv(kConv,tmp,net.layers{i}.w,net.layers{i}.stride,'forward');
            end
            
            if net.layers{i}.outputMaps ~= 1
                net.layers{i}.ReLUout = zeros(size(net.layers{i}.ReLUin),'single');
                net.layers{i + 1}.input = zeros(size(net.layers{i}.ReLUout),'single');
                for j = 1:net.layers{i}.outputMaps
                    if strcmp(train_or_test,'train')
                        [net.layers{i}.ReLUout(:,:,:,:,j),tmp_mean_mu, tmp_mean_sigma2 ]...
                            = my3DBatchNormalization(net.layers{i}.ReLUin(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                            net.layers{i}.beta(j,1),'forward',0,train_or_test,0,0);
                        
                        net.layers{i}.mean_mu(j,1) = net.layers{i}.mean_mu(j,1) + tmp_mean_mu;
                        net.layers{i}.mean_sigma2(j,1) = net.layers{i}.mean_sigma2(j,1) + tmp_mean_sigma2;
                    elseif strcmp(train_or_test,'test')
                        net.layers{i}.ReLUout(:,:,:,:,j) = my3DBatchNormalization(net.layers{i}.ReLUin(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                            net.layers{i}.beta(j,1),'forward',0,train_or_test,net.layers{i}.mean_mu(j,1),net.layers{i}.mean_sigma2(j,1));
                    end
                    net.layers{i+1}.input(:,:,:,:,j) = myLeakyReLU(net.layers{i}.ReLUout(:,:,:,:,j),lReLU_rate,'forward',0);
                end 
            else
                net.layers{i+1}.input = net.layers{i}.ReLUin;
            end
        elseif strcmp(net.layers{i}.type,'output')
            net.layers{i}.output = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.output = mySigmoidFun(net.layers{i}.input,'forward',0);
        end
        
    end
    fprintf('finished forward loop in discriminator %s\n',datestr(now,13));
elseif strcmp(forward_or_backward,'backward')
    %% for Discriminator bp, but when update is false, don't update weights of the network
    
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,1);
    batchSizeForCompute = 100;
    
    for i = numel(net.layers):-1:1
        if strcmp(net.layers{i}.type,'fullconnect')
            %there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution')
            if net.layers{i}.outputMaps ~= 1
                net.layers{i}.dBN = zeros(size(net.layers{i+1}.dinput),'single');
                net.layers{i}.dReLU = zeros(size(net.layers{i+1}.dinput),'single');

                for j=1:net.layers{i}.outputMaps
                    net.layers{i}.dBN(:,:,:,:,j) = myLeakyReLU(net.layers{i}.ReLUout(:,:,:,:,j),lReLU_rate,'backward',net.layers{i+1}.dinput(:,:,:,:,j));
                    
                    if strcmp(train_or_test,'train')
                        [net.layers{i}.dReLU(:,:,:,:,j),net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)] = ...
                            my3DBatchNormalization(net.layers{i}.ReLUin(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                            net.layers{i}.beta(j,1),'backward',net.layers{i}.dBN(:,:,:,:,j),train_or_test,0,0);
                    elseif strcmp(train_or_test,'test')
                        net.layers{i}.dReLU(:,:,:,:,j) = my3DBatchNormalization(net.layers{i}.ReLUin(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                            net.layers{i}.beta(j,1),'backward',net.layers{i}.dBN(:,:,:,:,j),...
                            train_or_test,net.layers{i}.mean_mu(j,1),net.layers{i}.mean_sigma2(j,1));
                    end

%                     [net.layers{i}.dBN(:,:,:,:,j),net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)] = ...
%                         my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),net.layers{i}.lamda(j,1),...
%                         net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput(:,:,:,:,j));
%                     
%                     net.layers{i}.dReLU(:,:,:,:,j) = myLeakyReLU(net.layers{i}.ReLUin(:,:,:,:,j),lReLU_rate,'backward',net.layers{i}.dBN(:,:,:,:,j));
                end
            else
                net.layers{i}.dReLU = net.layers{i + 1}.dinput;
            end
            
            % compute dx
            if size(net.layers{i+1}.dinput, 5)==1
                kConv=kConv_backward_r;
            elseif size(net.layers{i}.w, 5) ~= 1
                kConv=kConv_backward_my;
            elseif size(net.layers{i}.w, 5) == 1
                kConv=kConv_backward;
            end
            
            tmp = zeros([batch_size,size(net.layers{i}.input,2) + 2 * net.layers{i}.padding,...
                size(net.layers{i}.input,3) + 2 * net.layers{i}.padding,...
                size(net.layers{i}.input,4) + 2 * net.layers{i}.padding,size(net.layers{i}.input,5)],'single');
            
            if(batch_size > batchSizeForCompute)
                iterate_num = ceil(size(net.layers{i}.input,1)/batchSizeForCompute);
                for tmp_x=1:iterate_num
                    if tmp_x == iterate_num
                        start_num = (tmp_x-1) * batchSizeForCompute + 1;
                        end_num = size(net.layers{i}.input,1);
                    else
                        start_num = (tmp_x-1) * batchSizeForCompute + 1;
                        end_num = tmp_x * batchSizeForCompute;
                    end
                    
                    tmpin = zeros(end_num-start_num,size(net.layers{i}.dReLU,2),size(net.layers{i}.dReLU,3),...
                        size(net.layers{i}.dReLU,4),size(net.layers{i}.dReLU,5),'single');                
                    for tmp_batch = start_num:end_num
                        tmpin(tmp_batch-(tmp_x-1) * batchSizeForCompute,:,:,:,:)=net.layers{i}.dReLU(tmp_batch,:,:,:,:);
                    end
                    
                    tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'backward');

                    for tmp_batch = start_num:end_num
                        tmp(tmp_batch,:,:,:,:) = tmpout(tmp_batch - (tmp_x-1) * batchSizeForCompute,:,:,:,:);
                    end
                end
            else
                tmp = myGPUConv(kConv,net.layers{i}.dReLU,net.layers{i}.w,net.layers{i}.stride,'backward');
            end
            
            net.layers{i}.dinput = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.dinput = tmp(:,1 + net.layers{i}.padding:end - net.layers{i}.padding,1 + net.layers{i}.padding:end - net.layers{i}.padding,...
                1 + net.layers{i}.padding:end - net.layers{i}.padding,:);
            
            %fprintf('finished %dth backpropagation loop for dx in discriminator %s\n',i,datestr(now,13));
            
            % compute dw
            if strcmp(update,'true')
                net.layers{i}.dw=net.layers{i}.dw.*0;
                
                tmp = zeros([batch_size,size(net.layers{i}.input,2) + 2 * net.layers{i}.padding,...
                    size(net.layers{i}.input,3)+ 2 * net.layers{i}.padding,...
                    size(net.layers{i}.input,4)+ 2 * net.layers{i}.padding,size(net.layers{i}.input,5)],'single');
                tmp(:,1+net.layers{i}.padding:end-net.layers{i}.padding,1+net.layers{i}.padding:end-net.layers{i}.padding,...
                    1+net.layers{i}.padding:end-net.layers{i}.padding,:) = net.layers{i}.input;
                
                if size(net.layers{i + 1}.dinput,5) == 1
                    kConv_w = kConv_weight_r;
                elseif size(net.layers{i}.w, 5) ~= 1
                    kConv_w = kConv_weight_c;
                elseif size(net.layers{i}.w, 5) == 1
                    kConv_w = kConv_weight;
                end
                
                if(batch_size > batchSizeForCompute)
                    iterate_num = ceil(size(net.layers{i}.input,1)/batchSizeForCompute);
                    for tmp_x=1:iterate_num
                        if tmp_x == iterate_num
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = size(net.layers{i}.input,1);
                        else
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = tmp_x * batchSizeForCompute;
                        end
                        
                        tmpdin = zeros(end_num-start_num,size(net.layers{i+1}.dinput,2),size(net.layers{i+1}.dinput,3),...
                            size(net.layers{i+1}.dinput,4),size(net.layers{i+1}.dinput,5),'single');
                        tmpin = zeros(end_num-start_num,size(tmp,2),size(tmp,3),...
                            size(tmp,4),...
                            size(tmp,5),'single');
                        for tmp_batch = start_num:end_num
                            tmpin(tmp_batch - (tmp_x-1) * batchSizeForCompute,:,:,:,:) = tmp(tmp_batch,:,:,:,:);
                            tmpdin(tmp_batch - (tmp_x-1) * batchSizeForCompute,:,:,:,:) = net.layers{i+1}.dinput(tmp_batch,:,:,:,:);
                        end
                         
                        tmpout = myGPUConv(kConv_w,tmpin,tmpdin,net.layers{i}.stride,'weight');
                        
                        net.layers{i}.dw =net.layers{i}.dw + tmpout;
                    end
                else
                    net.layers{i}.dw = myGPUConv(kConv_w,tmp,net.layers{i+1}.dinput,net.layers{i}.stride,'weight');
                end
                
                %fprintf('finished %dth backpropagation loop for dw in discriminator %s\n',i,datestr(now,13));
            end
        elseif strcmp(net.layers{i}.type,'output')
            net.layers{i}.dinput = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.dinput= mySigmoidFun(net.layers{i}.input,'backward',x);
        end
    end
    
    %calc gradient for every weigths by using Nesterov momentum algorithm
    if strcmp(update,'true')
        momentum = net.momentum;
        momentum2 = net.momentum2;
        lr = net.lr;
        BN_lr = net.BNlr;
        for i=1:(numel(net.layers)-1)
            %ascending the discriminator loss
            %net.layers{i}.histdw2 = momentum2 * net.layers{i}.histdw2 + (1-momentum2).*net.layers{i}.dw;
            net.layers{i}.histdw = momentum * net.layers{i}.histdw + (1-momentum).*net.layers{i}.dw.^2;
            net.layers{i}.w = net.layers{i}.w - lr.*(net.layers{i}.dw)./(sqrt(net.layers{i}.histdw)+1.0e-8);
            
            for j=1:net.layers{i}.outputMaps
                net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1)-BN_lr.*net.layers{i}.dlamda(j,1);
                net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1)-BN_lr.*net.layers{i}.dbeta(j,1);
            end
        end
        fprintf('finished a gradient calculate procedure in discriminator %s\n',datestr(now,13)); 
    end    
end
    y=net;
end

