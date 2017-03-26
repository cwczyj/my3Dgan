function y = initialMeanBN( net )
%INITIALMEANBN Summary of this function goes here
%   Detailed explanation goes here

for i=1:numel(net.layers)-1
    net.layers{i}.mean_mu = zeros([net.layers{i}.outputMaps,1],'single');
    net.layers{i}.mean_sigma2 = zeros([net.layers{i}.outputMaps,1],'single');
end

y = net;

end

