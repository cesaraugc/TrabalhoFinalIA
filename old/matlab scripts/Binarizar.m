%Saída: Binariza a imagem segmentada
%Entrada: function X = Binarizar(name)
%Retorna uma imagem binarizada
%Autora Carolina Watanabe Silva

function X = Binarizar(name)

seg = imread(name);
% label2rgb(bwlabel(seg));
% RGB = label2rgb(bwlabel(seg));
% imshow(RGB)
[c,l] = size(seg);
c = int16(c/2);
l = int16(l/2);
seg(:)=seg(:)+1;   %  é necessário porque a imagem indexada inicia no zero
i = seg(c,l);
J = seg;
ind = find(seg(:)~=i);    %escolhe a classe 5
J(ind) = 0;
map(1,:) = [0,0,0];
map(i,:) = [1,1,1];
X = ind2rgb(J,map);  %Transforma a imagem indexada para rgb
X = im2bw(X);   %binariza a imagem
X = Region(X);
%imshow(X)