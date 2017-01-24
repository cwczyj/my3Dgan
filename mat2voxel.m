%convert a mesh data (mat flie) to voxel

fileFolder=fullfile('ModelNet40mat');
dirFolder=dir(fullfile(fileFolder,'*'));
folderNames={dirFolder.name}';

for i=3:length(folderNames)
    dirOut=dir(fullfile([fileFolder,'/',char(folderNames(i)),'/train'],'*.mat'));
    filenames={dirOut.name}';
    
    mkdir(['ModelNet40Voxel','/',char(folderNames(i)),'/train']);
    for j=1:length(filenames)
        load([fileFolder,'/',char(folderNames(i)),'/train','/',char(filenames(j))]); 
        vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
        FV.faces = faces;
        FV.vertices = vertices;
        Volume=polygon2voxel(FV,[64 64 64],'auto');
        [pathstr,name, ext] = fileparts(char(filenames(j)));
        save(['ModelNet40Voxel','/',char(folderNames(i)),'/train','/',name,'.mat'],'Volume');
    end
end