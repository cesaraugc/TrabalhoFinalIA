%Esta função escolhe a região a ser analisada
%Retorna uma imagem binarizada
%Autora Carolina Watanabe da Silva

function L=Region(X);

[n,p] = size(X);
n = round(n/2);
p = round(p/2);

%y = X(n,p);

[L,num] = bwlabel(X);

L = L+1;
y = L(n,p);
%num = uint8(num);

if num>1
    for i=1:num
        map(i,:) = [0,0,0];
    end
    %map(1:num,:) = [0,0,0];
    map(y,:) = [1,1,1];
    L = ind2rgb(L,map);
    L = im2bw(L);
else
    L = X;   
end

% J = L;
% 
% [L,num] = bwlabel(L);
% 
% if num>1
%     ind = find(L==2);
%     map(1,:) = [0,0,0];
%     map(2,:) = [1,1,1];
%     L = ind2rgb(L,map);
%     L = im2bw(L);
% else
%     L = J;   
% end