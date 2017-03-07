function [ y ] = myDiscriminator( net, x ,forward_or_backward , update )
%MYDISCRIMINATOR Summary of this function goes here
%   The network of discriminator
%   update is the flag for the net to judge wether to update weights in the
%   network;
%   x is the input of the network, when this is forwardfeed x is a batch of
%   voxel data (64x64x64x100); when this is backward x is a batch of the
%   Loss (1x100) for the GAN net, L = mean(log(D(x))+log(1-D(G(z))));

if strcmp(forward_or_backward,'forward')
    %% for Discriminator ff
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,4);
    net.layers{1}.layerSize = size(x,1);
    net.layers{1}.input{1} = x;
    
    for i=1:numel(net.layers)
        if strcmp(net.layers{i}.type,'fullconnect')
            % as if there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution')
            net.layers{i+1}.layerSize = (net.layers{i}.layerSize-net.layers{i}.kernels+2*net.layers{i}.padding)/net.layers{i}.stride+1;
            
            for j=1:net.layers{i}.outputMaps
                net.layers{i}.ReLUin{j}=zeros(net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                        batch_size);
                for k=1:batch_size
                    z = zeros(net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize);
                    for l=1:numel(net.layers{i}.input)
                        z = z + my3dConv(net.layers{i}.input{l}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
                            net.layers{i}.stride,net.layers{i}.padding,'C');
                    end
                    
                    net.layers{i}.ReLUin{j}(:,:,:,k)=z;
                end
                
                if strcmp(net.layers{i}.actFun,'LReLU')
                    net.layers{i}.ReLUout{j} = myLeakyReLU(net.layers{i}.ReLUin{j},lReLU_rate,'forward',0);                    
                end
                
                net.layers{i+1}.input{j} = my3DBatchNormalization...
                    (net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),net.layers{i}.beta(j,1),'forward',0);
            end
        elseif strcmp(net.layers{i}.type,'output')
            for j=1:numel(net.layers{i}.input)
                net.layers{i}.output{j} = mySigmoidFun(net.layers{i}.input{j},'forward',0);
            end
        end
        fprintf('finished on %dth forward loop in discriminator %s\n',i,datestr(now,13));
    end
    
elseif strcmp(forward_or_backward,'backward')
    %% for Discriminator bp, but when update is false, don't update weights of the network
    
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,4);
    for i = numel(net.layers):-1:1
        if strcmp(net.layers{i}.type,'fullconnect')
            %there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution')
            z = zeros(net.layers{i}.layerSize, net.layers{i}.layerSize, net.layers{i}.layerSize,...
                    numel(net.layers{i}.input),batch_size);
            
            %compute the dx
            for j=1:net.layers{i}.outputMaps
                net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
                [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
                        my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
                        net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
                    
                 if strcmp(net.layers{i}.actFun,'LReLU')
                     net.layers{i}.dReLU{j} = myLeakyReLU(net.layers{i}.ReLUin{j},lReLU_rate,'backward',net.layers{i}.dBN{j});     
                 end  
                    
                for k=1:batch_size
                    for l=1:numel(net.layers{i}.input)
                        z(:,:,:,l,k) = z(:,:,:,l,k) + my3dConv(net.layers{i}.dReLU{j}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
                            net.layers{i}.stride,net.layers{i}.padding,'T');
                    end
                end
            end
            
            %get the dx for the next layers.
            for j=1:numel(net.layers{i}.input)
                net.layers{i}.dinput{j}=z(:,:,:,j,:);
            end
            
            fprintf('finished %dth backpropagation loop for dx in discriminator %s\n',i,datestr(now,13));
            
            if strcmp(update,'true')
                net.layers{i}.dw=net.layers{i}.dw.*0;
                %compute the dw
                for j=1:net.layers{i}.outputMaps
                    % too important to get understand!!!
                    tmpSizeofInput = (size(net.layers{i+1}.dinput{j},1)-1)*(net.layers{i}.stride-1)+size(net.layers{i+1}.dinput{j},1);
                    for l=1:numel(net.layers{i}.input)    
                        for k = 1:batch_size
                            tmpInput = zeros(tmpSizeofInput,tmpSizeofInput,tmpSizeofInput);
                            tmpInput((1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end))= net.layers{i+1}.dinput{j}(:,:,:,k);
                            net.layers{i}.dw(:,:,:,l,j)=net.layers{i}.dw(:,:,:,l,j)+...
                                my3dConv(net.layers{i}.input{l}(:,:,:,k),tmpInput,1,net.layers{i}.padding,'C');
                        end
                    end
                end
                
                fprintf('finished %dth backpropagation loop for dw in discriminator %s\n',i,datestr(now,13));
            end
        elseif strcmp(net.layers{i}.type,'output')
            for j=1:numel(net.layers{i}.input)
                net.layers{i}.dinput{j} = mySigmoidFun(net.layers{i}.input{j},'backward',x);
            end
        end
    end
    
    %calc gradient for every weigths by using Nesterov momentum algorithm
    if strcmp(update,'true')
        momentum = net.momentum;
        lr = net.lr;
        BN_lr = net.BNlr;
%        wd = net.weight_decay;
        for i=1:(numel(net.layers)-1)
            %ascending the discriminator loss
%            net.layers{i}.histdw = momentum * net.layers{i}.histdw + lr * (net.layers{i}.dw + wd * net.layers{i}.w);
            net.layers{i}.histdw = momentum * net.layers{i}.histdw + (1-momentum).*net.layers{i}.dw.^2;
%            net.layers{i}.w = net.layers{i}.w - (net.layers{i}.histdw);
            net.layers{i}.w = net.layers{i}.w - lr.*(net.layers{i}.dw)./(sqrt(net.layers{i}.histdw)+1.0e-8);
            
            for j=1:net.layers{i}.outputMaps
%                 net.layers{i}.histdlamda(j,1) = momentum * net.layers{i}.histdlamda(j,1) + BN_lr * (net.layers{i}.dlamda(j,1) + wd*net.layers{i}.lamda(j,1));
%                 net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1) - (net.layers{i}.histdlamda(j,1));
%                 net.layers{i}.histdbeta(j,1) = momentum * net.layers{i}.histdbeta(j,1) + BN_lr * (net.layers{i}.dbeta(j,1) + wd*net.layers{i}.beta(j,1));
%                 net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1) - (net.layers{i}.histdbeta(j,1));

                net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1)-BN_lr.*net.layers{i}.dlamda(j,1);
                net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1)-BN_lr.*net.layers{i}.dbeta(j,1);
            end
        end
    end
    
    fprintf('finished a gradient calculate procedure in discriminator %s\n',datestr(now,13));    
end
    y=net;
end

