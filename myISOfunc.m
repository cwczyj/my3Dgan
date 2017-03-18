function [ output ] = myISOfunc( voxel, num )
%MYISOFUNC Summary of this function goes here
%   Detailed explanation goes here

sizeX = size(voxel,2);
sizeY = size(voxel,3);
sizeZ = size(voxel,4);

tmpISO = zeros(sizeX,sizeY,sizeZ,'single');
for i = 1:sizeX
    for j=1:sizeY
        for k=1:sizeZ
            tmpISO(i,j,k) = voxel(num, i, j, k);
        end
    end
end

isosurface(tmpISO);

end

