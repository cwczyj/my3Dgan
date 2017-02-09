function [ y ] = myGenerator( net, x ,forward_or_backward)
%MYGENERATOR Summary of this function goes here
%   The network of Generator in GAN
%   The parameter net is the structure of the Generator network;
%   and x is the batch of input of the network (200x100)
%   for ff; and x is the batch of loss of the network (64x64x64x100)for bp;
%   y is the output of the network(a [64 64 64] voxel)

    if strcmp(forward_or_backward,'forward')
    %% for Generator ff
        batch_size = size(x,2);
        net.layers{1}.input{1} = x;
        net.layers{1}.layerSize=200;
        net.layers{2}.layerSize = 4;
        
        for i = 1:(numel(net.layers)-1)

            if strcmp(net.layers{i}.type,'fullconnect')
                for j=1:net.layers{i}.outputMaps
                    z=zeros([net.layers{i}.kernels^3,batch_size]);
                    for k=1:numel(net.layers{i}.input)
                        z=z+net.layers{i}.w(:,:,j)*net.layers{i}.input{k};
                    end
                    net.layers{i}.ReLUin{j} = reshape(z,net.layers{i}.kernels,net.layers{i}.kernels,...
                        net.layers{i}.kernels,batch_size);
                    
                    net.layers{i}.ReLUout{j} = myReLU(net.layers{i}.ReLUin{j}, 'forward', 0);
                    net.layers{i+1}.input{j} = my3DBatchNormalization...
                        (net.layers{i}.ReLUout{j}, net.layers{i}.lamda(j,1), net.layers{i}.beta(j,1), 'forward',0);
                end
                
                fprintf('finish the %s layers\n',net.layers{i}.type);
            elseif strcmp(net.layers{i}.type,'convolution')
                net.layers{i+1}.layerSize=(net.layers{i}.layerSize-1)*(net.layers{i}.stride-1)+...
                                           net.layers{i}.layerSize+net.layers{i}.kernels-1-2;
                for j=1:net.layers{i}.outputMaps             
                    net.layers{i}.ReLUin{j}=zeros(net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                        batch_size);
                    for k=1:batch_size
                        z=zeros(net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize);
                        for l=1:numel(net.layers{i}.input)
                            z=z+my3dConv(net.layers{i}.input{l}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
                                net.layers{i}.stride,1,'T');
                        end
                        net.layers{i}.ReLUin{j}(:,:,:,k)=z;
                    end
                    
                    if strcmp(net.layers{i}.actFun,'ReLU')
                        net.layers{i}.ReLUout{j} = myReLU(net.layers{i}.ReLUin{j},'forward',0);
                    elseif strcmp(net.layers{i}.actFun,'sigmoid')
                        net.layers{i}.ReLUout{j} = mySigmoidFun(net.layers{i}.ReLUin{j},'forward',0);
                    end
                    
                    %there is no Batch Normaliaztion before the last
                    %layer
                    if i~=5
                        net.layers{i+1}.input{j} = my3DBatchNormalization...
                            (net.layers{i}.ReLUout{j}, net.layers{i}.lamda(j,1), net.layers{i}.beta(j,1),'forward',0);
                    else
                        net.layers{i+1}.input{j}=net.layers{i}.ReLUout{j};
                    end 
                end
            end
             fprintf('finished the %dth layer in generator %s\n',i,datestr(now,13));
        end      
        
    elseif strcmp(forward_or_backward,'backward')
        %% for Generator bp
        
        batch_size=size(x,5);
        net.layers{6}.dinput{1}=x;
        for i=(numel(net.layers)-1):-1:1
            if strcmp(net.layers{i}.type,'fullconnect')
                for j=1:net.layers{i}.outputMaps
                    if i~=5
                        net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
                        [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
                                my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
                                net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
                    else
                        net.layers{i+1}.dinput{j} = reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
                        net.layers{i}.dBN{j} = net.layers{i+1}.dinput{j};
                    end
                    
                    net.layers{i}.dReLU{j} = myReLU(net.layers{i}.ReLUin{j},'backward',net.layers{i}.dBN{j});
                    
                    tmp = reshape(net.layers{i}.dReLU{j},net.layers{i}.kernels^3,size(net.layers{i}.dReLU{j},4));
                    for k=1:numel(net.layers{i}.input)
                        net.layes{i}.dw(:,:,j)=tmp*net.layers{i}.input{k}';
                    end
                end
            elseif strcmp(net.layers{i}.type,'convolution')
                 z=zeros(net.layers{i}.layerSize,net.layers{i}.layerSize,net.layers{i}.layerSize,...
                     numel(net.layers{i}.input),batch_size);
                
                %compute dx
                for j=1:net.layers{i}.outputMaps
                    if i~= 5
                        net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
                        [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
                            my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
                            net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
                    else
                        net.layers{i+1}.dinput{j} = reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
                        net.layers{i}.dBN{j} = net.layers{i+1}.dinput{j};
                    end
                    
                    if strcmp(net.layers{i}.actFun,'ReLU')
                        net.layers{i}.dReLU{j} = myReLU(net.layers{i}.ReLUin{j},'backward',net.layers{i}.dBN{j});
                    elseif strcmp(net.layers{i}.actFun,'sigmoid')
                        net.layers{i}.dReLU{j} = mySigmoidFun(net.layers{i}.ReLUin{j},'backward',net.layers{i}.dBN{j});
                    end
                    
                    for k=1:batch_size
                        for l=1:numel(net.layers{i}.input)
                            z(:,:,:,l,k) = z(:,:,:,l,k) + my3dConv(net.layers{i}.dReLU{j}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
                                net.layers{i}.stride,net.layers{i}.padding,'C');
                        end
                    end
                end
                
                for j=1:numel(net.layers{i}.input)
                    net.layers{i}.dinput{j}=z(:,:,:,j,:);
                end
                
                fprintf('finish %dth bp layer for dx in generator at %s\n',i,datestr(now,13));
                
                %compute dw
                for l=1:numel(net.layers{i}.input)
                    tmpSizeofInput = (size(net.layers{i}.input{l},1)-1)*(net.layers{i}.stride-1)+size(net.layers{i}.input{l},1);
                    for j=1:net.layers{i}.outputMaps
                        for k = 1:batch_size
                            tmpInput = zeros(tmpSizeofInput,tmpSizeofInput,tmpSizeofInput);
                            tmpInput((1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end))= net.layers{i}.input{l}(:,:,:,k);
                            net.layers{i}.dw(:,:,:,l,j)=net.layers{i}.dw(:,:,:,l,j)+...
                                my3dConv(net.layers{i+1}.dinput{j}(:,:,:,k),tmpInput,1,net.layers{i}.padding,'C');
                        end
                        
                        net.layers{i}.dw=net.layers{i}.dw/batch_size;
                    end
                end                
            end
            
            fprintf('finished %dth bp layer in generator %s\n',i,datestr(now,13));
        end
        
        momentum = net.momentum;
        lr = net.lr;
        BN_lr = net.BNlr;
        wd = net.weight_decay;
        %calc gradient for every weigths by using Nesterov momentum algorithm.
        for i=1:(numel(net.layers)-1)
            net.layers{i}.histdw = momentum * net.layers{i}.histdw + lr * (net.layers{i}.dw + wd * net.layers{i}.w);
            net.layers{i}.w = net.layers{i}.w - (net.layers{i}.histdw);
            
            if i ~= 5
                for j=1:net.layers{i}.outputMaps
                    net.layers{i}.histdlamda(j,1) = momentum * net.layers{i}.histdlamda(j,1) + BN_lr * (net.layers{i}.dlamda(j,1) + net.layers{i}.lamda(j,1));
                    net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1) - (net.layers{i}.histdlamda(j,1));

                    net.layers{i}.histdbeta(j,1) = momentum * net.layers{i}.histdbeta(j,1) + BN_lr * (net.layers{i}.dbeta(j,1) + net.layers{i}.beta(j,1));
                    net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1) - (net.layers{i}.histdbeta(j,1));
                end
            end
        end
        
        fprintf('finished a gradient calculate procedure in generator %s\n',datestr(now,13));
    end
    
    y=net;
end

