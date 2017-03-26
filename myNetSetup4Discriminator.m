function [ net ] = myNetSetup4Discriminator( net , fanin)
%MYNETSETUP Summary of this function goes here
%   initial the weights of a network, revise for GPU computation;
    numlayers = numel(net.layers)-1;
    fan_in = fanin;
    
    for i=1:numlayers
        
        fan_out = net.layers{i}.outputMaps;
        Guass_std = sqrt(2.0/(net.layers{i}.kernels^3*(fan_in+fan_out)));
            
        if strcmp(net.layers{i}.type,'fullconnect')
            %15 article for ReLU networks 
            %(Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification)
            net.layers{i}.w=normrnd(0,Guass_std,[fan_in, net.layers{i}.kernels^3, net.layers{i}.outputMaps]);
            net.layers{i}.w=single(net.layers{i}.w);
            net.layers{i}.dw=zeros([net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps],'single');
            net.layers{i}.histdw=zeros([net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps],'single');
            net.layers{i}.histdw2=zeros([net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps],'single');
        elseif strcmp(net.layers{i}.type,'convolution')
            net.layers{i}.w=normrnd(0,Guass_std,[fan_out,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in]);
            net.layers{i}.w=single(net.layers{i}.w);
            net.layers{i}.dw=zeros([fan_out,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in],'single');
            net.layers{i}.histdw=zeros([fan_out,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in],'single');
            net.layers{i}.histdw2=zeros([fan_out,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in],'single');
        end
        %lamda and beta for Batch Normalization layer;
        net.layers{i}.lamda = (rand([net.layers{i}.outputMaps,1],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out));
        net.layers{i}.beta = (rand([net.layers{i}.outputMaps,1],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out));
        
        net.layers{i}.dlamda = zeros([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.dbeta = zeros([net.layers{i}.outputMaps,1],'single');
        
        net.layers{i}.mean_mu = zeros([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.mean_sigma2 = zeros([net.layers{i}.outputMaps,1],'single');
        
        
        fan_in = net.layers{i}.outputMaps;
    end
end

