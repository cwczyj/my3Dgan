function [ output ] = myConcatenate( x, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if size(x,2) ~= size(y,2) && size(x,3) ~= size(y,3) && size(x,4) ~= size(y,4)
    fprintf('Data Error!/n');
end

sizeX = size(x,1);
sizeY = size(y,1);

output = zeros([sizeX + sizeY,size(x,2),size(x,3),size(x,4)],'single');

output(1:sizeX,:,:,:) = x;
output(sizeX + 1:sizeX + sizeY,:,:,:) = y;


end

