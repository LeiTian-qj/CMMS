function min_idx = searchBestIndicator(Z,F,C)
% search clustering indicator vector
%----------------------------------------------------------------------------------------------------------
obj = zeros(C,1);
for j = 1:C       
    obj(j,1) = (norm(Z-F(:,j)))^2; 
end
[~,min_idx] = min(obj); 
end
