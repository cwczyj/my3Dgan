function [ y ] = myGenerator( net, x ,forward_or_backward)
%MYGENERATOR Summary of this function goes here
%   The network of Generator in GAN
%   The parameter net is the structure of the Generator network;
%   and x is the batch of input of the network (200x100)
%   for ff; and x is the batch of loss of the network (64x64x64x100)for bp;
%   y is the output of the network(a [64 64 64] voxel)


global kConv_backward kConv_forward2 kConv_backward_c kConv_forward kConv_forward_c kConv_weight kConv_weight_c kConv_backward_my;

    if strcmp(forward_or_backward,'forward')
    %% for Generator ff
        batch_size = size(x,1);
        net.layers{1}.input = x;
        net.layers{1}.layerSize=200;
        net.layers{2}.layerSize = 4;
        batchSizeForCompute = 50;
        
        for i = 1:numel(net.layers)

            if strcmp(net.layers{i}.type,'fullconnect')
                net.layers{i+1}.input=zeros([batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i}.outputMaps],'single');
                net.layers{i}.ReLUin=zeros([batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i}.outputMaps],'single');
                net.layers{i}.ReLUout=zeros([batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i}.outputMaps],'single');
                
                for j=1:net.layers{i}.outputMaps
                    z=zeros([batch_size,net.layers{i}.kernels^3],'single');
                    z=z+net.layers{i}.input*net.layers{i}.w(:,:,j);
                    
                    net.layers{i}.ReLUin(:,:,:,:,j) = reshape(z,batch_size,net.layers{i}.kernels,...
                        net.layers{i}.kernels,net.layers{i}.kernels);
                    
                    net.layers{i}.ReLUout(:,:,:,:,j) = myReLU(net.layers{i}.ReLUin(:,:,:,:,j), 'forward', 0);
                    net.layers{i+1}.input(:,:,:,:,j) = my3DBatchNormalization...
                        (net.layers{i}.ReLUout(:,:,:,:,j), net.layers{i}.lamda(j,1), net.layers{i}.beta(j,1), 'forward',0);
                end
            elseif strcmp(net.layers{i}.type,'convolution')
                net.layers{i+1}.layerSize=(net.layers{i}.layerSize-1)*(net.layers{i}.stride-1)+...
                    net.layers{i}.layerSize+net.layers{i}.kernels-1-2*net.layers{i}.padding;
                
                net.layers{i}.ReLUtmpin=zeros([batch_size,net.layers{i+1}.layerSize+2*net.layers{i}.padding...
                    ,net.layers{i+1}.layerSize+2*net.layers{i}.padding,...
                    net.layers{i+1}.layerSize+2*net.layers{i}.padding,net.layers{i}.outputMaps],'single');
                net.layers{i}.ReLUin=zeros([batch_size,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i}.outputMaps],'single');
                
                if size(net.layers{i}.w,5)==1
                    kConv=kConv_backward;
                else
                    kConv=kConv_backward_my;
                end
                
                if(size(net.layers{i}.input,1) > batchSizeForCompute)
                    iterate_num = ceil(size(net.layers{i}.input,1)/batchSizeForCompute);
                    for tmp_x=1:iterate_num
                        if tmp_x == iterate_num
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = size(net.layers{i}.input,1);
                        else
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = tmp_x * batchSizeForCompute;
                        end
                        
                        tmpin = zeros(end_num-start_num,size(net.layers{i}.input,2),size(net.layers{i}.input,3),...
                            size(net.layers{i}.input,4),size(net.layers{i}.input,5),'single');                      
                        for tmp_batch = start_num:end_num
                            tmpin(tmp_batch,:,:,:,:)=net.layers{i}.input(tmp_batch,:,:,:,:);
                        end
                        
                        tmpout = zeros(end_num-start_num,size(net.layers{i}.ReLUtmpin,2),size(net.layers{i}.ReLUtmpin,3),...
                            size(net.layers{i}.ReLUtmpin,4),size(net.layers{i}.ReLUtmpin,5),'single');  
                        tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'backward');
                        
                        for tmp_batch = start_num:end_num
                            net.layers{i}.ReLUtmpin(tmp_batch,:,:,:,:) = tmpout(tmp_batch,:,:,:,:);
                        end
                    end
                else
                    net.layers{i}.ReLUtmpin = myGPUConv(kConv,net.layers{i}.input,net.layers{i}.w,net.layers{i}.stride,'backward');
                end
            
                net.layers{i}.ReLUin=net.layers{i}.ReLUtmpin(:,1+net.layers{i}.padding:end-net.layers{i}.padding,...
                    1+net.layers{i}.padding:end-net.layers{i}.padding,1+net.layers{i}.padding:end-net.layers{i}.padding,:);
                
                net.layers{i}.ReLUout=zeros([batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i}.outputMaps],'single');
                net.layers{i+1}.input=zeros([batch_size,net.layers{i+1}.layerSize,net.layers{i+1}.layerSize,...
                    net.layers{i+1}.layerSize,net.layers{i}.outputMaps],'single');
                
                for j=1:net.layers{i}.outputMaps
                    net.layers{i}.ReLUout(:,:,:,:,j)=myReLU(net.layers{i}.ReLUin(:,:,:,:,j),'forward',0);
                    net.layers{i+1}.input(:,:,:,:,j)=my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),...
                        net.layers{i}.lamda(j,1),net.layers{i}.beta(j,1),'forward',0);
                end
                                       
            elseif strcmp(net.layers{i}.type,'output')
                net.layers{i}.output=zeros(size(net.layers{i}.input),'single');               
                for j=1:size(net.layers{i}.input,5)
                    net.layers{i}.output(:,:,:,:,j)=mySigmoidFun(net.layers{i}.input(:,:,:,:,j),'forward',0);
                end
            end
             fprintf('finished the %dth layer in generator %s\n',i,datestr(now,13));
        end      
        
    elseif strcmp(forward_or_backward,'backward')
        %% for Generator bp
        
        batch_size=size(x,1);
        batchSizeForCompute = 50;
        for i=numel(net.layers):-1:1
            if strcmp(net.layers{i}.type,'fullconnect')
                for j=1:net.layers{i}.outputMaps
                    net.layers{i}.dBN(:,:,:,:,j) = zeros(size(net.layers{i+1}.dinput(:,:,:,:,j)),'single');
                    [net.layers{i}.dBN(:,:,:,:,j),net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
                        my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                        net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput(:,:,:,:,j));
                    
                    net.layers{i}.dReLU(:,:,:,:,j) = myReLU(net.layers{i}.ReLUin(:,:,:,:,j),'backward',net.layers{i}.dBN(:,:,:,:,j));
                    
                    tmp = reshape(net.layers{i}.dReLU(:,:,:,:,j),size(net.layers{i}.dReLU(:,:,:,:,j),1),...
                        net.layers{i}.kernels^3);
                    net.layers{i}.dw(:,:,j) = net.layers{i}.input' * tmp;
                end
            elseif strcmp(net.layers{i}.type,'convolution')
                %   compute dx
                net.layers{i}.dBN = zeros(size(net.layers{i+1}.dinput),'single');
                net.layers{i}.dReLU = zeros(size(net.layers{i+1}.dinput),'single');
                
                for j=1:net.layers{i}.outputMaps
                    [net.layers{i}.dBN(:,:,:,:,j),net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)] = ...
                        my3DBatchNormalization(net.layers{i}.ReLUout(:,:,:,:,j),net.layers{i}.lamda(j,1),...
                        net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput(:,:,:,:,j));
                    
                    net.layers{i}.dReLU(:,:,:,:,j) = myReLU(net.layers{i}.ReLUin(:,:,:,:,j),'backward',net.layers{i}.dBN(:,:,:,:,j));
                end
                
                tmp = zeros([size(net.layers{i}.dReLU,1),size(net.layers{i}.dReLU,2)+2 * net.layers{i}.padding,...
                    size(net.layers{i}.dReLU,3)+2 * net.layers{i}.padding,size(net.layers{i}.dReLU,4)+2 * net.layers{i}.padding,...
                    size(net.layers{i}.dReLU,5)],'single');
                tmp(1+net.layers{i}.padding:end-net.layers{i}.padding,1+net.layers{i}.padding:end-net.layers{i}.padding,...
                    1+net.layers{i}.padding:end-net.layers{i}.padding) = net.layers{i}.dReLU;
                
                net.layers{i}.dinput = zeros(batch_size,net.layers{i}.layerSize,net.layers{i}.layerSize,net.layers{i}.layerSize,...
                    size(net.layers{i}.input,5));
                
                if size(tmp,5)==1
                    kConv=kConv_forward;
                else
                    kConv=kConv_forward_c;
                end
                
                if(size(net.layers{i}.input,1) > batchSizeForCompute)
                    iterate_num = ceil(size(net.layers{i}.input,1)/batchSizeForCompute);
                    for tmp_x=1:iterate_num
                        if tmp_x == iterate_num
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = size(net.layers{i}.input,1);
                        else
                            start_num = (tmp_x-1) * batchSizeForCompute + 1;
                            end_num = tmp_x * batchSizeForCompute;
                        end
                        
                        tmpin = zeros(end_num-start_num,size(net.layers{i}.input,2),size(net.layers{i}.input,3),...
                            size(net.layers{i}.input,4),size(net.layers{i}.input,5),'single');                      
                        for tmp_batch = start_num:end_num
                            tmpin(tmp_batch,:,:,:,:)=tmp(tmp_batch,:,:,:,:);
                        end
                        
                        tmpout = zeros(end_num-start_num,size(net.layers{i}.ReLUtmpin,2),size(net.layers{i}.ReLUtmpin,3),...
                            size(net.layers{i}.ReLUtmpin,4),size(net.layers{i}.ReLUtmpin,5),'single');  
                        tmpout = myGPUConv(kConv,tmpin,net.layers{i}.w,net.layers{i}.stride,'forward');
                        
                        for tmp_batch = start_num:end_num
                            net.layers{i}.ReLUtmpin(tmp_batch,:,:,:,:) = tmpout(tmp_batch,:,:,:,:);
                        end
                    end
                else
                    net.layers{i}.dinput = myGPUConv(kConv,tmp,net.layers{i}.w,net.layers{i}.stride,'forward');
                end

                % compute dw
            elseif strcmp(net.layers{i}.type,'output')
                net.layers{i}.dinput = zeros(size(net.layers{i}.input),'single');
                x=reshape(x,size(net.layers{i}.input));
                net.layers{i}.dinput = mySigmoidFun(net.layers{i}.input,'backward',x);
            end
%             if strcmp(net.layers{i}.type,'fullconnect')
%                 for j=1:net.layers{i}.outputMaps
%                     net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
%                     [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
%                             my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
%                             net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
%                     
%                     net.layers{i}.dReLU{j} = myReLU(net.layers{i}.ReLUin{j},'backward',net.layers{i}.dBN{j});
%                     
%                     tmp = reshape(net.layers{i}.dReLU{j},net.layers{i}.kernels^3,size(net.layers{i}.dReLU{j},4));
%                     for k=1:numel(net.layers{i}.input)
%                         net.layers{i}.dw(:,:,j)=tmp*net.layers{i}.input{k}';
%                     end
%                 end
%             elseif strcmp(net.layers{i}.type,'convolution')
%                  z=zeros(net.layers{i}.layerSize,net.layers{i}.layerSize,net.layers{i}.layerSize,...
%                      numel(net.layers{i}.input),batch_size);
%                 
%                 %compute dx
%                 for j=1:net.layers{i}.outputMaps
%                     net.layers{i+1}.dinput{j}=reshape(net.layers{i+1}.dinput{j},size(net.layers{i}.ReLUout{j}));
%                     [net.layers{i}.dBN{j},net.layers{i}.dlamda(j,1),net.layers{i}.dbeta(j,1)]=...
%                         my3DBatchNormalization(net.layers{i}.ReLUout{j},net.layers{i}.lamda(j,1),...
%                         net.layers{i}.beta(j,1),'backward',net.layers{i+1}.dinput{j});
%                     
%                     if strcmp(net.layers{i}.actFun,'ReLU')
%                         net.layers{i}.dReLU{j} = myReLU(net.layers{i}.ReLUin{j},'backward',net.layers{i}.dBN{j});                
%                     end
%                     
%                     for k=1:batch_size
%                         for l=1:numel(net.layers{i}.input)
%                             z(:,:,:,l,k) = z(:,:,:,l,k) + my3dConv(net.layers{i}.dReLU{j}(:,:,:,k),net.layers{i}.w(:,:,:,l,j),...
%                                 net.layers{i}.stride,net.layers{i}.padding,'C');
%                         end
%                     end
%                 end
%                 
%                 for j=1:numel(net.layers{i}.input)
%                     net.layers{i}.dinput{j}=z(:,:,:,j,:);
%                 end
%                 
%                 fprintf('finish %dth bp layer for dx in generator at %s\n',i,datestr(now,13));
%                 
%                 %compute dw
%                 net.layers{i}.dw=net.layers{i}.dw.*0;
%                 for l=1:numel(net.layers{i}.input)
%                     tmpSizeofInput = (size(net.layers{i}.input{l},1)-1)*(net.layers{i}.stride-1)+size(net.layers{i}.input{l},1);
%                     for j=1:net.layers{i}.outputMaps
%                         for k = 1:batch_size
%                             tmpInput = zeros(tmpSizeofInput,tmpSizeofInput,tmpSizeofInput);
%                             tmpInput((1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end),(1:net.layers{i}.stride:end))= net.layers{i}.input{l}(:,:,:,k);
%                             net.layers{i}.dw(:,:,:,l,j)=net.layers{i}.dw(:,:,:,l,j)+...
%                                 my3dConv(net.layers{i+1}.dinput{j}(:,:,:,k),tmpInput,1,net.layers{i}.padding,'C');
%                         end
%                     end
%                 end
%                 
%             elseif strcmp(net.layers{i}.type,'output')
%                 for j=1:numel(net.layers{i}.input)
%                     x=reshape(x,size(net.layers{i}.input{j}));
%                     net.layers{i}.dinput{j} = mySigmoidFun(net.layers{i}.input{j},'backward',x);
%                 end
%             end
%             
%             fprintf('finished %dth bp layer in generator %s\n',i,datestr(now,13));
        end
        
        %ascending the discriminator loss
        momentum = net.momentum;
        lr = net.lr;
        BN_lr = net.BNlr;
        for i=1:(numel(net.layers)-1)
             net.layers{i}.histdw = momentum * net.layers{i}.histdw + (1-momentum).*net.layers{i}.dw.^2;
             net.layers{i}.w = net.layers{i}.w - lr.*(net.layers{i}.dw)./(sqrt(net.layers{i}.histdw)+1.0e-8);

            
            for j=1:net.layers{i}.outputMaps
                net.layers{i}.lamda(j,1) = net.layers{i}.lamda(j,1)-BN_lr.*net.layers{i}.dlamda(j,1);
                net.layers{i}.beta(j,1) = net.layers{i}.beta(j,1)-BN_lr.*net.layers{i}.dbeta(j,1);
            end
        end
        
        fprintf('finished a gradient calculate procedure in generator %s\n',datestr(now,13));
    end
    
    y=net;
end

