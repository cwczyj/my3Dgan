function [ y ] = myDiscriminator( net, x ,forward_or_backward , update )
%MYDISCRIMINATOR Summary of this function goes here
%   The network of discriminator
%   update is the flag for the net to judge wether to update weights in the
%   network;
%   x is the input of the network, when this is forwardfeed x is a batch of
%   voxel data (64x64x64x100); when this is backward x is a batch of the
%   Loss (1x100) for the GAN net, L = mean(log(D(x))+log(1-D(G(z))));

global kConv_backward kConv_forward kConv_forward_c kConv_weight kConv_weight_c kConv_backward_my;

if strcmp(forward_or_backward,'forward')
    %% for Discriminator ff
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,1);
    net.layers{1}.layerSize = size(x,1);
    net.layers{1}.input = x;
    batchSizeForCompute = 50;
    
    for i=1:numel(net.layers)
        if strcmp(net.layers{i}.type,'fullconnect')
            % as if there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution') 
            %forward   
            net.layers{i+1}.layerSize = (net.layers{i}.layerSize-net.layers{i}.kernels+2*net.layers{i}.padding)/net.layers{i}.stride+1;
            
            net.layers{i}.ReLUin = zeros(batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                net.layers{i+1}.layerSize,net.layers{i}.outputMaps,'single');
            
            if size(net.layers{i}.input,5) == 1
                kConv = kConv_forward;
            else
                kConv = kConv_forward_c;
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
                        tmpin(tmp_batch,:,:,:,:)=tmp((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:);
                    end
                    
                    tmpout = zeros(end_num-start_num,size(net.layers{i}.ReLUtmpin,2),size(net.layers{i}.ReLUtmpin,3),...
                        size(net.layers{i}.ReLUtmpin,4),size(net.layers{i}.ReLUtmpin,5),'single');
                    tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'forward');

                    for tmp_batch = start_num:end_num
                        net.layers{i}.ReLUin((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:) = tmpout(tmp_batch,:,:,:,:);
                    end
                end
            else
                net.layers{i}.ReLUin = myGPUConv(kConv,tmp,net.layers{i}.w,net.layers{i}.stride,'forward');
            end
            
            net.layers{i}.ReLUout = zeros(size(net.layers{i}.ReLUin),'single');
            net.layers{i + 1}.input = zeros(size(net.layers{i}.ReLUout),'single');
            for j = 1:net.layers{i}.outputMaps
                net.layers{i}.ReLUout(:,:,:,:,j) = myLeakyReLU(net.layers{i}.ReLUin(:,:,:,:,j),lReLU_rate,'forward',0);
                net.layers{i+1}.input(:,:,:,:,j) = my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                    net.layers{i}.beta(j,1),'forward',0);
            end
        elseif strcmp(net.layers{i}.type,'output')
            net.layers{i}.output = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.output = mySigmoidFun(net.layers{i}.input,'forward',0);
        end
        fprintf('finished on %dth forward loop in discriminator %s\n',i,datestr(now,13));
    end
    
elseif strcmp(forward_or_backward,'backward')
    %% for Discriminator bp, but when update is false, don't update weights of the network
    
    lReLU_rate = net.LeakyReLU;
    batch_size = size(x,1);
    batchSizeForCompute = 50;
    
    for i = numel(net.layers):-1:1
        if strcmp(net.layers{i}.type,'fullconnect')
            %there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution')
            net.layers{i}.dBN = zeros(size(net.layers{i+1}.dinput),'single');
            net.layers{i}.dReLU = zeros(size(net.layers{i+1}.dinput),'single');
                
            for j=1:net.layers{i}.outputMaps
                [net.layers{i}.dBN(:,:,:,:,j),net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)] = ...
                    my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                    net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput(:,:,:,:,j));
                    
                net.layers{i}.dReLU(:,:,:,:,j) = myLeakyReLU(net.layers{i}.ReLUin(:,:,:,:,j),lReLU_rate,'backward',net.layers{i}.dBN(:,:,:,:,j));
            end
            
            % compute dx
            if size(net.layers{i}.dinput,5)==1
                kConv=kConv_backward;
            else
                kConv=kConv_backward_my;
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
                        tmpin(tmp_batch,:,:,:,:)=net.layers{i}.dReLU((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:);
                    end
                    
                    tmpout = zeros(end_num-start_num,size(net.layers{i}.ReLUtmpin,2),size(net.layers{i}.ReLUtmpin,3),...
                        size(net.layers{i}.ReLUtmpin,4),size(net.layers{i}.ReLUtmpin,5),'single');
                    tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'backward');

                    for tmp_batch = start_num:end_num
                        tmp((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:) = tmpout(tmp_batch,:,:,:,:);
                    end
                end
            else
                tmp = myGPUConv(kConv,net.layers{i}.dReLU,net.layers{i}.w,net.layers{i}.stride,'backward');
            end
            
            net.layers{i}.dinput = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.dinput = tmp(:,1 + net.layers{i}.padding:end - net.layers{i}.padding,1 + net.layers{i}.padding:end - net.layers{i}.padding,...
                1 + net.layers{i}.padding:end - net.layers{i}.padding,:);
            
            fprintf('finished %dth backpropagation loop for dx in discriminator %s\n',i,datestr(now,13));
            
            % compute dw
            if strcmp(update,'true')
                net.layers{i}.dw=net.layers{i}.dw.*0;
                
                tmp = zeros([batch_size,size(net.layers{i}.dinput,2) + 2 * net.layers{i}.padding,...
                    size(net.layers{i}.dinput,3)+ 2 * net.layers{i}.padding,...
                    size(net.layers{i}.dinput,4)+ 2 * net.layers{i}.padding,size(net.layers{i}.dinput,5)],'single');
                tmp(:,1+net.layers{i}.padding:end-net.layers{i}.padding,1+net.layers{i}.padding:end-net.layers{i}.padding,...
                    1+net.layers{i}.padding:end-net.layers{i}.padding,:) = net.layers{i}.dinput;
                
                if size(net.layers{i}.dinput,5) == 1
                    kConv_w = kConv_weight;
                else
                    kConv_w = kConv_weight_c;
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
                        
                        tmpdin = zeros(end_num-start_num,size(net.layers{i}.dinput,2),size(net.layers{i}.dinput,3),...
                            size(net.layers{i}.dinput,4),size(net.layers{i}.dinput,5),'single');
                        tmpin = zeros(end_num-start_num,size(net.layers{i}.input,2),size(net.layers{i}.input,3),...
                            size(net.layers{i}.input,4),...
                            size(net.layers{i}.input,5),'single');
                        for tmp_batch = start_num:end_num
                            tmpdin(tmp_batch,:,:,:,:) = tmp((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:);
                            tmpin(tmp_batch,:,:,:,:) = net.layers{i}.input((tmp_x-1) * batchSizeForCompute + tmp_batch,:,:,:,:);
                        end
                        
                        tmpout = zeros(end_num-start_num,size(net.layers{i}.dw,2),size(net.layers{i}.dw,3),size(net.layers{i}.dw,4),...
                            size(net.layers{i}.dw,5),'single');  
                        tmpout = myGPUConv(kConv_w,tmpdin,tmpin,net.layers{i}.stride,'weight');
                        
                        net.layers{i}.dw =net.layers{i}.dw + tmpout;
                    end
                else
                    net.layers{i}.dw = myGPUConv(kConv_w,tmp,net.layers{i}.input,net.layers{i}.stride,'weight');
                end
                
                fprintf('finished %dth backpropagation loop for dw in discriminator %s\n',i,datestr(now,13));
            end
%             z = zeros(net.layers{i}.layerSize, net.layers{i}.layerSize, net.layers{i}.layerSize,...
%                     numel(net.layers{i}.input),batch_size);
%             
%             %compute the dx
%             for j=1:net.layers{i}.outputMaps
%                 net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
%                 [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
%                         my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
%                         net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
%                     
%                  if strcmp(net.layers{i}.actFun,'LReLU')
%                      net.layers{i}.dReLU{j} = myLeakyReLU(net.layers{i}.ReLUin{j},lReLU_rate,'backward',net.layers{i}.dBN{j});     
%                  end  
%                     
%                 for k=1:batch_size
%                     for l=1:numel(net.layers{i}.input)
%                         z(:,:,:,l,k) = z(:,:,:,l,k) + my3dConv(net.layers{i}.dReLU{j}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
%                             net.layers{i}.stride,net.layers{i}.padding,'T');
%                     end
%                 end
%             end
%             
%             %get the dx for the next layers.
%             for j=1:numel(net.layers{i}.input)
%                 net.layers{i}.dinput{j}=z(:,:,:,j,:);
%             end
%             
%             fprintf('finished %dth backpropagation loop for dx in discriminator %s\n',i,datestr(now,13));
%             
%             if strcmp(update,'true')
%                 net.layers{i}.dw=net.layers{i}.dw.*0;
%                 %compute the dw
%                 for j=1:net.layers{i}.outputMaps
%                     % too important to get understand!!!
%                     tmpSizeofInput = (size(net.layers{i+1}.dinput{j},1)-1)*(net.layers{i}.stride-1)+size(net.layers{i+1}.dinput{j},1);
%                     for l=1:numel(net.layers{i}.input)    
%                         for k = 1:batch_size
%                             tmpInput = zeros(tmpSizeofInput,tmpSizeofInput,tmpSizeofInput);
%                             tmpInput((1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end))= net.layers{i+1}.dinput{j}(:,:,:,k);
%                             net.layers{i}.dw(:,:,:,l,j)=net.layers{i}.dw(:,:,:,l,j)+...
%                                 my3dConv(net.layers{i}.input{l}(:,:,:,k),tmpInput,1,net.layers{i}.padding,'C');
%                         end
%                     end
%                 end
%                 
%                 fprintf('finished %dth backpropagation loop for dw in discriminator %s\n',i,datestr(now,13));
%             end
        elseif strcmp(net.layers{i}.type,'output')
            net.layers{i}.dinput = zeros(size(net.layers{i}.input),'single');
            net.layers{i}.dinput= mySigmoidFun(net.layers{i}.input,'backward',x);
        end
    end
    
    %calc gradient for every weigths by using Nesterov momentum algorithm
    if strcmp(update,'true')
        momentum = net.momentum;
        lr = net.lr;
        BN_lr = net.BNlr;
        for i=1:(numel(net.layers)-1)
            %ascending the discriminator loss
            net.layers{i}.histdw = momentum * net.layers{i}.histdw + (1-momentum).*net.layers{i}.dw.^2;
            net.layers{i}.w = net.layers{i}.w - lr.*(net.layers{i}.dw)./(sqrt(net.layers{i}.histdw)+1.0e-8);
            
            for j=1:net.layers{i}.outputMaps
                net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1)-BN_lr.*net.layers{i}.dlamda(j,1);
                net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1)-BN_lr.*net.layers{i}.dbeta(j,1);
            end
        end
    end
    
    fprintf('finished a gradient calculate procedure in discriminator %s\n',datestr(now,13));    
end
    y=net;
end

