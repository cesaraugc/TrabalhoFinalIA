%Cria as máscaras
dirname = 'D:\usuarios\carolina\carolina\Doutorado\Programas\Base_mama\JPG\segmentation5c';
files = dir(strcat(dirname, '\*bmp'));     % pega o nome das imagens
for i=1:250
    nameimage = fullfile(dirname,files(i).name);
    I = Binarizar(nameimage);
    imwrite(I,files(i).name);
end
    
    