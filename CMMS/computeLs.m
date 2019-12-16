function [Ls] = computeLs(Ys,C)
ns = length(Ys);
SClassNum = zeros(C,1);
for c = reshape(unique(Ys),1,C)
    SClassNum(c)=length(find(Ys==c));
end
BBS1=zeros(ns,ns);
for i=1:ns
    BBS1(i,i) = 1.0 - 1.0/SClassNum(Ys(i));
    for j=(i+1):ns
        if Ys(i)==Ys(j)
            BBS1(i,j) = 1.0/SClassNum(Ys(i))*(-1);
            BBS1(j,i) = 1.0/SClassNum(Ys(i))*(-1);
        end
    end
end
Ls = BBS1/norm(BBS1,'fro');
end