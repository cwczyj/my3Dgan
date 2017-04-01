function [ net ] = myNetSetup( net , fanin)
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
        elseif strcmp(net.layers{i}.type,'convolution')
            net.layers{i}.w=(rand([fan_in,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_out],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out));
            net.layers{i}.b=(rand([fan_out,1],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out)); 
            net.layers{i}.dw=zeros([fan_in,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_out],'single');
            net.layers{i}.histdw=zeros([fan_in,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_out],'single');
            net.layers{i}.histdw2=zeros([fan_in,net.layers{i}.kernels, net.layers{i}.kernels, net.layers{i}.kernels,...
                fan_out],'single');
            net.layers{i}.db=zeros([fan_out,1],'single');
            net.layers{i}.histdb=zeros([fan_out,1],'single');
            net.layers{i}.histdb2=zeros([fan_out,1],'single');
        end
        %lamda and beta for Batch Normalization layer;
        net.layers{i}.lamda = (rand([net.layers{i}.outputMaps,1],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out));
        net.layers{i}.beta = (rand([net.layers{i}.outputMaps,1],'single')-0.5)*2 * sqrt(6 / (fan_in + fan_out));
        
        net.layers{i}.dlamda = zeros([net.layers{i}.outputMaps,1],'single');
        net.layers{i}.dbeta = zeros([net.layers{i}.outputMaps,1],'single');
        
        fan_in = net.layers{i}.outputMaps;
    end
end

