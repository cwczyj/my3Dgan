function [ output ] = myLeakyReLU( A ,rate, forward_or_backward, Loss )
%MYLEAKYRELU Summary of this function goes here
%   LeakyReLU

    if strcmp(forward_or_backward,'forward')
        tmp = (A<0);
        tmp = tmp*rate;
        output= tmp.*A + max(A,0);
    elseif strcmp(forward_or_backward,'backward')
        tmp = (A<0);
        tmp = (tmp*rate).*Loss;
        tmp2 = (A>0);
        tmp2 = tmp2.*Loss;
        output = tmp + tmp2;
    end

end

