%Read batch of off mesh data, and convert them to voxel grid data

fileFolder=fullfile('ModelNet40');
dirFolder=dir(fullfile(fileFolder,'*'));
folderNames={dirFolder.name}';

for i=3:length(folderNames)
    dirOut=dir(fullfile([fileFolder,'/',char(folderNames(i)),'/train'],'*.off'));
    filenames={dirOut.name}';
    
    mkdir(['ModelNet40mat','/',char(folderNames(i)),'/train']);
    for j=1:length(filenames)
        offObj=off_loader([fileFolder,'/',char(folderNames(i)),'/train','/',char(filenames(j))],0,'x',1);
        [pathstr,name, ext] = fileparts(char(filenames(j)));
        vertices=offObj.vertices;
        faces=offObj.faces;
        save(['ModelNet40mat','/',char(folderNames(i)),'/train','/',name,'.mat'],'vertices','faces');
    end
end