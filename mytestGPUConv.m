global kConv_backward kConv_forward2 kConv_backward_c kConv_forward kConv_forward_c kConv_weight kConv_weight_c kConv_backward_my;

rng('shuffle');

%% test kConv_forward2;
if 0
    data=rand(2,3,3,3,'single');
    kernel=rand(16,2,2,2,'single');
    stride=1;
    kConv=kConv_forward2;
    kConv1=kConv_forward;

    numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4); 
    numFilters = size(kernel,1); filterSize =  size(kernel,2); 

    paddingStart = 0; moduleStride = stride; imgStride = numImages; numGroups = 1;
    numModulesX = (imgSizeX - filterSize) / stride + 1; numModulesY = (imgSizeY - filterSize) / stride + 1; numModulesZ = (imgSizeZ - filterSize) / stride + 1;

    filterPerThread = 4; imagePerThread = 1;

    kConv.ThreadBlockSize = [4, 32];
    kConv.GridSize = [numModulesX * numModulesY * numModulesZ * numFilters / (filterPerThread * 4), ceil(numImages/(32 * imagePerThread))];

    target = zeros([numImages, numModulesX, numModulesY, numModulesZ, numFilters], 'single');

    tic
    target_gpu = feval(kConv,...
        target, data, kernel,...
        numImages, numFilters, imgSizeZ, imgSizeY, imgSizeX, filterSize, ...
        paddingStart, moduleStride, numModulesZ, numModulesY, numModulesX, imgStride);

    target = gather(target_gpu);
    toc
    
    kConv1.ThreadBlockSize = [4, 32];
    kConv1.GridSize = [numModulesX * numModulesY * numModulesZ * numFilters / (filterPerThread * 4), ceil(numImages/(32 * imagePerThread))];
    
    target1 = zeros([numImages, numModulesX, numModulesY, numModulesZ, numFilters], 'single');

    tic
    target_gpu = feval(kConv1,...
        target1, data, kernel,...
        numImages, numFilters, imgSizeZ, imgSizeY, imgSizeX, filterSize, ...
        paddingStart, moduleStride, numModulesZ, numModulesY, numModulesX, imgStride);

    target1 = gather(target_gpu);
    toc

    originalConv=zeros(size(target),'single');
    tmpkernel=zeros(size(kernel),'single');
    for i=1:size(kernel,1)
        tmpkernel(i,:,:,:)=flip(flip(flip(kernel(i,:,:,:), 2), 3),4);
    end
    tic
    for i=1:size(data,1)
        for j=1:size(kernel,1)
            originalConv(i,:,:,:,j) = originalConv(i,:,:,:,j) +convn(data(i,:,:,:),tmpkernel(j,:,:,:),'valid');
        end
    end
    toc

    tmpdata=zeros([size(data,2),size(data,3),size(data,4),size(data,1)],'single');
    for i=1:size(data,1)
        tmpdata(:,:,:,i)=data(i,:,:,:);
    end

    tmpkernel=zeros([size(kernel,2),size(kernel,3),size(kernel,4),size(kernel,1)],'single');
    for j=1:size(kernel,1)
        tmpkernel(:,:,:,j)=kernel(j,:,:,:);
    end

    MyConv=zeros([size(target,2),size(target,3),size(target,4),size(target,1),size(target,5)],'single');
    tic
    for i=1:size(data,1)
        for j=1:size(kernel,1)
           MyConv(:,:,:,i,j) = MyConv(:,:,:,i,j)+my3dConv(tmpdata(:,:,:,i),tmpkernel(:,:,:,j),stride,paddingStart,'C');
        end
    end
    toc

    MyFinConv=zeros(size(target),'single');
    for i=1:size(data,1)
        for j=1:size(kernel,1)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end
%% test kConv_forward_c;
if 0
    data=rand([2,8,8,8,10],'single');
    kernel=rand([32,4,4,4,10],'single');
    stride=1;
    numColors=size(data,5);
    kConv=kConv_forward_c;

    numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4); 
    numFilters = size(kernel,1); filterSize =  size(kernel,2); 

    paddingStart = 0; moduleStride = stride; imgStride = numImages; numGroups = 1;
    numModulesX = (imgSizeX - filterSize) / stride + 1; numModulesY = (imgSizeY - filterSize) / stride + 1; numModulesZ = (imgSizeZ - filterSize) / stride + 1;

    filterPerThread = 8; imagePerThread = 1;

    kConv.ThreadBlockSize = [4,32];
    kConv.GridSize = [numModulesX * numModulesY * numModulesZ * numFilters / (filterPerThread * 4), ceil(numImages/(32 * imagePerThread))];

    target = zeros(numImages, numModulesX, numModulesY, numModulesZ, numFilters, 'single');

    tic
    target_gpu = feval(kConv,...
        target, data, kernel,...
        numImages, numFilters, imgSizeZ, imgSizeY, imgSizeX, filterSize, ...
        paddingStart, moduleStride, numModulesZ, numModulesY, numModulesX, imgStride,numColors, numGroups);

    target = gather(target_gpu);
    toc
    
%     originalConv=zeros(size(target));
%     tmpkernel=zeros(size(kernel));
%     for i=1:size(kernel,1)
%         for j=1:size(kernel,5)
%             tmpkernel(i,:,:,:,j)=flip(flip(flip(kernel(i,:,:,:,j), 2), 3),4);
%         end
%     end
%     tic
%     for i=1:size(data,1)
%         for j=1:size(kernel,1)
%             for k=1:size(data,5)
%                 originalConv(i,:,:,:,j) = originalConv(i,:,:,:,j) +convn(data(i,:,:,:,k),tmpkernel(j,:,:,:,k),'valid');
%             end
%         end
%     end
%     toc
    
    tmpdata=zeros(size(data,2),size(data,3),size(data,4),size(data,1),size(data,5));
    for i=1:size(data,1)
        for j=1:size(data,5)
            tmpdata(:,:,:,i,j)=data(i,:,:,:,j);
        end
    end

    tmpkernel=zeros(size(kernel,2),size(kernel,3),size(kernel,4),size(kernel,1),size(kernel,5));
    for j=1:size(kernel,1)
        for k=1:size(kernel,5)
            tmpkernel(:,:,:,j,k)=kernel(j,:,:,:,k);
        end
    end

    MyConv=zeros(size(target,2),size(target,3),size(target,4),size(target,1),size(target,5));
    tic
    for i=1:size(data,1)
        for j=1:size(kernel,1)
            for k=1:size(data,5)
                MyConv(:,:,:,i,j) = MyConv(:,:,:,i,j)+my3dConv(tmpdata(:,:,:,i,k),tmpkernel(:,:,:,j,k),stride,paddingStart,'C');
            end
        end
    end
    toc

    MyFinConv=zeros(size(target));
    for i=1:size(data,1)
        for j=1:size(kernel,1)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end

%% test kConv_weight_c
if 0
    data=rand(2,5,5,5,16);
    kernel=rand(2,4,4,4,32);
    stride=1;
    numColors=size(data,5);
    kConv=kConv_weight_c;
    
    numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4);
    numFilters = size(kernel,5); numModulesX = size(kernel,2); numModulesY = size(kernel,3); numModulesZ = size(kernel,4);
    paddingStart = 0; moduleStride = stride; imgStride = numImages; partialSum = numModulesX * numModulesY * numModulesZ;
    filterSize = imgSizeX - stride * (numModulesX - 1);

    preLoadCases = 32; filtersPerThread = 2; colorsPerThread = 8;
    scaleOutput = 1; numGroups = 1;

    kConv.ThreadBlockSize = [16, 8];
    kConv.GridSize = [numFilters*numModulesX*numModulesY*numModulesZ/partialSum/16/filtersPerThread, ceil(filterSize^3 / 8) * (numColors / colorsPerThread)];

    target = zeros(numFilters, filterSize, filterSize, filterSize, numColors, 'single');
    
    tic
    target_gpu = feval(kConv, ....
        target, data, kernel,...
        numImages, numFilters, numModulesZ, numModulesY, numModulesX, imgSizeZ, imgSizeY, imgSizeX, ...
        filterSize, paddingStart, moduleStride, imgStride, numColors, numGroups, partialSum, scaleOutput);

    target = gather(target_gpu);
    toc
    
    
    tmpdata=zeros(size(data,2),size(data,3),size(data,4),size(data,1),size(data,5));
    for i=1:size(data,1)
        for j=1:size(data,5)
            tmpdata(:,:,:,i,j)=data(i,:,:,:,j);
        end
    end

    tmpkernel=zeros(size(kernel,2),size(kernel,3),size(kernel,4),size(kernel,1),size(kernel,5));
    for j=1:size(kernel,1)
        for k=1:size(kernel,5)
            tmpkernel(:,:,:,j,k)=kernel(j,:,:,:,k);
        end
    end
    
    MyConv=zeros(size(target,2),size(target,3),size(target,4),size(target,1),size(target,5));
    tic
    for i=1:size(data,1)
        for k=1:size(kernel,5)
            for j=1:size(data,5)
                MyConv(:,:,:,k,j)=MyConv(:,:,:,k,j)+my3dConv(tmpdata(:,:,:,i,j),tmpkernel(:,:,:,i,k),stride,paddingStart,'C');
            end
        end
    end
    toc
    
    MyFinConv=zeros(size(target));
    for i=1:size(target,1)
        for j=1:size(target,5)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end

%% test kConv_weight
if 1
    data=randi([1,10],100,10,10,10,'single');
    kernel=randi([1,10],100,4,4,4,16,'single');
    stride=1;
    numColors=size(data,5);
    kConv=kConv_weight;
    
    numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4);
    numFilters = size(kernel,5); numModulesX = size(kernel,2); numModulesY = size(kernel,3); numModulesZ = size(kernel,4);
    paddingStart = 0; moduleStride = stride; imgStride = numImages; partialSum = numModulesX * numModulesY * numModulesZ;
    filterSize = imgSizeX - stride * (numModulesX - 1);

    pixelsPerThread = 5; filtersPerThread = 2; colorsPerThread = 8;
    scaleOutput = 1; numGroups = 1;
    
    kConv.ThreadBlockSize = [16, 8];
    kConv.GridSize = [numFilters*numModulesX*numModulesY*numModulesZ/partialSum/16, ceil(filterSize^3 /(8*pixelsPerThread))];

    target = zeros(numFilters, filterSize, filterSize, filterSize, numColors, 'single');
    
    tic
    target_gpu = feval(kConv, ....
        target, data, kernel,...
        numImages, numFilters, numModulesZ, numModulesY, numModulesX, imgSizeZ, imgSizeY, imgSizeX, ...
        filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleOutput);

    target = gather(target_gpu);
    toc
    
    
    tmpdata=zeros(size(data,2),size(data,3),size(data,4),size(data,1),size(data,5));
    for i=1:size(data,1)
        for j=1:size(data,5)
            tmpdata(:,:,:,i,j)=data(i,:,:,:,j);
        end
    end
    
    tmpsize=stride*(size(kernel,2)-1)+1;
    tmpkernel=zeros(tmpsize,tmpsize,tmpsize,size(kernel,1),size(kernel,5));
    for j=1:size(kernel,1)
        for k=1:size(kernel,5)
            tmpkernel((1:stride:end),(1:stride:end),(1:stride:end),j,k)=kernel(j,:,:,:,k);
        end
    end
    
    MyConv=zeros(size(target,2),size(target,3),size(target,4),size(target,1),size(target,5));
    tic
    for i=1:size(data,1)
        for k=1:size(kernel,5)
            for j=1:size(data,5)
                MyConv(:,:,:,k,j)=MyConv(:,:,:,k,j)+my3dConv(tmpdata(:,:,:,i,j),tmpkernel(:,:,:,i,k),stride,paddingStart,'C');
            end
        end
    end
    toc
    
    MyFinConv=zeros(size(target));
    for i=1:size(target,1)
        for j=1:size(target,5)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end

%% test kConv_backward_c
if 0
    data=randi([1,10],32,2,2,2,16,'single');
    kernel=randi([1,10],16,3,3,3,16,'single');
    stride=1;
    numColors=size(kernel,5);
    kConv=kConv_backward_c;
    
    numImages = size(data,1); numModulesX = size(data,2); numModulesY = size(data,3); numModulesZ = size(data,4); moduleStride = stride;
    numFilters = size(kernel,1); filterSize = size(kernel,2);

    imgSizeX = stride * (numModulesX - 1) + filterSize; imgSizeY = stride * (numModulesY - 1) + filterSize; imgSizeZ = stride * (numModulesZ - 1) + filterSize;
    paddingStart = 0; numGroups = 1;

    colorsPerThread = 4; imgsPerThread = 1;

    kConv.ThreadBlockSize = [32, 4];
    kConv.GridSize = [ceil(numImages/(imgsPerThread * 32)) * (numColors / (4 * colorsPerThread)), imgSizeZ * imgSizeY * imgSizeX];

    target = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
    
    tic
    target_gpu = feval(kConv,....
        target, data, kernel,...
        numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
        imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride, numColors, numGroups);
    toc
    
    target = gather(target_gpu);
    
        
    tmpdata=zeros(size(data,2),size(data,3),size(data,4),size(data,1),size(data,5));
    for i=1:size(data,1)
        for j=1:size(data,5)
            tmpdata(:,:,:,i,j)=data(i,:,:,:,j);
        end
    end

    tmpkernel=zeros(size(kernel,2),size(kernel,3),size(kernel,4),size(kernel,1),size(kernel,5));
    for j=1:size(kernel,1)
        for k=1:size(kernel,5)
            tmpkernel(:,:,:,j,k)=kernel(j,:,:,:,k);
        end
    end
    
    MyConv = zeros(imgSizeZ,imgSizeX,imgSizeY,size(data,1),size(kernel,5));
    tic
    for i=1:size(data,1)
        for j=1:size(data,5)
            for k=1:size(kernel,5)
                MyConv(:,:,:,i,k)=MyConv(:,:,:,i,k)+my3dConv(tmpdata(:,:,:,i,j),tmpkernel(:,:,:,j,k),stride,paddingStart,'T');
            end
        end
    end
    toc
    
    MyFinConv=zeros(size(target));
    for i=1:size(target,1)
        for j=1:size(target,5)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end

%% test kConv_back
if 0
    data=rand(2,2,2,2,16,'single');
    kernel=rand(16,3,3,3,'single');
    stride=1;
    numColors=size(kernel,5);
    kConv=kConv_backward;
    
    numImages = size(data,1); numModulesX = size(data,2); numModulesY = size(data,3); numModulesZ = size(data,4); moduleStride = stride;
    numFilters = size(kernel,1); filterSize = size(kernel,2); 

    imgSizeX = stride * (numModulesX - 1) + filterSize; imgSizeY = stride * (numModulesY - 1) + filterSize; imgSizeZ = stride * (numModulesZ - 1) + filterSize;
    paddingStart = 0; imgsPerThread = 2;

    kConv.ThreadBlockSize = [16, 16];
    kConv.GridSize = [ceil(numImages/(imgsPerThread *16)), imgSizeZ * ceil(imgSizeY/4) * ceil(imgSizeX/4)];

    target = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
    tic
    target_gpu = feval(kConv,...
        target, data, kernel,...
        numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
        imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride);
    toc
    
    target = gather(target_gpu);
    
    tmpdata=zeros(size(data,2),size(data,3),size(data,4),size(data,1),size(data,5),'single');
    for i=1:size(data,1)
        for j=1:size(data,5)
            tmpdata(:,:,:,i,j)=data(i,:,:,:,j);
        end
    end

    tmpkernel=zeros(size(kernel,2),size(kernel,3),size(kernel,4),size(kernel,1),size(kernel,5),'single');
    for j=1:size(kernel,1)
        for k=1:size(kernel,5)
            tmpkernel(:,:,:,j,k)=kernel(j,:,:,:,k);
        end
    end
    
    MyConv = zeros(imgSizeZ,imgSizeX,imgSizeY,size(data,1),size(kernel,5),'single');
    tic
    for i=1:size(data,1)
        for j=1:size(data,5)
            for k=1:size(kernel,5)
                MyConv(:,:,:,i,k)=MyConv(:,:,:,i,k)+my3dConv(tmpdata(:,:,:,i,j),tmpkernel(:,:,:,j,k),stride,paddingStart,'T');
            end
        end
    end
    toc
    
    MyFinConv=zeros(size(target),'single');
    for i=1:size(target,1)
        for j=1:size(target,5)
           MyFinConv(i,:,:,:,j) = MyConv(:,:,:,i,j);
        end
    end
end

%% test kConv_backward_my
if 0
    data=rand(25,4,4,4,512,'single');
    kernel=rand(512,4,4,4,16,'single');
    stride=1;
    numColors=size(kernel,5);
    kConv=kConv_backward_my;
    
    numImages = size(data,1); numModulesX = size(data,2); numModulesY = size(data,3); numModulesZ = size(data,4); moduleStride = stride;
    numFilters = size(kernel,1); filterSize = size(kernel,2); 

    imgSizeX = stride * (numModulesX - 1) + filterSize; imgSizeY = stride * (numModulesY - 1) + filterSize; imgSizeZ = stride * (numModulesZ - 1) + filterSize;
    paddingStart = 0; numGroups = 1;
    
    colorsPerThread = 4; imgsPerThread = 1;

    kConv.ThreadBlockSize = [32, 16];
    kConv.GridSize = [ceil(numImages/(imgsPerThread * 32)) * numColors, imgSizeZ * ceil(imgSizeY/4) * ceil(imgSizeX/4)];

    target = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
    tic
    target_gpu = feval(kConv,...
        target, data, kernel,...
        numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
        imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride, numColors, numGroups);
    toc
    
    target = gather(target_gpu);

    kConv1 = kConv_backward_c;
    kConv1.ThreadBlockSize = [32, 4];
    kConv1.GridSize = [ceil(numImages/(imgsPerThread * 32)) * (numColors / (4 * colorsPerThread)) , imgSizeZ * imgSizeY * imgSizeX];
    
    target1 = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
    
    tic
    target_gpu1 = feval(kConv1,....
        target1, data, kernel,...
        numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
        imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride, numColors, numGroups);
    toc
    
    target1 = gather(target_gpu1);
end