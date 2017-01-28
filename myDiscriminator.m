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
    
    for i=1:(numel(net.layers)-1)
        if strcmp(net.layers{i}.type,'fullconnect')
            % as if there is no fullconnect layer in the discriminator net.
        elseif strcmp(net.layers{i}.type,'convolution')
            
        end
    end
elseif strcmp(forward_or_backward,'backward')
    %% for Discriminator bp, but when update is false, don't update weights of the network
    
end

end

