function y = getMeanBN( net, num )
%GETMEANBN Summary of this function goes here
%   Detailed explanation goes here
for i=1:numel(net.layers)-1
        net.layers{i}.mean_mu = net.layers{i}.mean_mu./num;
        net.layers{i}.mean_sigma2 = net.layers{i}.mean_sigma2./num;
end

y = net;

end

