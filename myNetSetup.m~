function [ net ] = myNetSetup( net )
%MYNETSETUP Summary of this function goes here
%   initial the weights of a network
    numlayers = numel(net.layers)-1;
    fan_in = 200;
    
    for i=1:numlayers
        
        fan_out = net.layers{i}.outputMaps;
        Guass_std = sqrt(2.0/(net.layers{i}.kernels^3*(fan_in+fan_out)));
            
        if strcmp(net.layers{i}.type,'fullconnect')
            %15 article for ReLU networks 
            %(Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification)
            net.layers{i}.w=normrnd(0,Guass_std,[net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps]);
            net.layers{i}.dw=zeros([net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps]);
            net.layers{i}.histdw=zeros([net.layers{i}.kernels^3, fan_in, net.layers{i}.outputMaps]);
        elseif strcmp(net.layers{i}.type,'convolution')
            net.layers{i}.w=normrnd(0,Guass_std,[net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in, fan_out]);
            net.layers{i}.dw=zeros([net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in, fan_out]);
            net.layers{i}.histdw=zeros([net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_in, fan_out]);
        end
        %lamda and beta for Batch Normalization layer;
        net.layers{i}.lamda = rand([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.beta = rand([net.layers{i}.outputMaps,1],'single');
        
        net.layers{i}.dlamda = zeros([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.dbeta = zeros([net.layers{i}.outputMaps,1],'single');
        
        net.layers{i}.histdlamda = zeros([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.histdbeta = zeros([net.layers{i}.outputMaps,1],'single');
        
        fan_in = net.layers{i}.outputMaps;
    end
end

