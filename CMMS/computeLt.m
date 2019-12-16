function [Lt] = computeLt(Zt,k,r)
nt = size(Zt,2);
distx = L2_distance_1(Zt,Zt);
[~, idx] = sort(distx,2);
S = zeros(nt,nt);
for j =1:nt
    idxa0 = idx(j,2:k+1);
    dxj = distx(j,idxa0);
    ad = -dxj/(2*r);
    S(j,idxa0) = EProjSimplex_new(ad);
end
S = (S+S')/2;
Lt = diag(sum(S,2))-S;
Lt = Lt/norm(Lt,'fro');
end