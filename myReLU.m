function [ output ] = myReLU( A , forward_or_backward, Loss )
%MYRELU Summary of this function goes here
%   ReLU layers for 3D gan, A is a batch of voxel matrix;
%   Loss is a batch of loss data;

    if strcmp(forward_or_backward,'forward')
        output=max(A,0);
    elseif strcmp(forward_or_backward,'backward')
        tmp=(A>0);
        output=Loss.*tmp;
    end


end

