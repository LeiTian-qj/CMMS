function [Acc] = CMMS_lg(Xs,Xt,Ys,Yt,options)

if ~isfield(options,'d')
    options.d = 100;
end
if ~isfield(options,'alpha')
    options.alpha = 0.1;
end
if ~isfield(options,'beta')
    options.beta = 0.1;
end
if ~isfield(options,'gamma')
    options.gamma = 5.0;
end
if ~isfield(options,'k')
    options.k = 10;
end
if ~isfield(options,'T')
    options.T = 10;
end
d = options.d;
alpha = options.alpha;
beta = options.beta;
gamma = options.gamma;
k = options.k;
T = options.T;

% Set predefined variables
[m,ns] = size(Xs);
[~,nt] = size(Xt);
X = [Xs,Xt];
n = ns + nt;

X = X*diag((1./sqrt(sum(X.^2))));
Xss = X(:,1:ns);
Xtt = X(:,ns+1:end);
C = length(unique(Ys));

T1 = X*X'-(X*ones(n,1))*(ones(1,n)/n*X');
Es = zeros(ns,C);
for c = reshape(unique(Ys),1,C)
    Es(find(Ys==c),c) = 1/length(find(Ys==c));
end
E = [Es;zeros(nt,C)];

%%compute Ls
Ls = computeLs(Ys,C);
%%compute Lt
distXt = L2_distance_1(X(:,ns+1:end),X(:,ns+1:end));
[distX1, idx] = sort(distXt,2);
W = zeros(nt);
rr = zeros(nt,1);
for i = 1:nt
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
r = mean(rr);
W = (W+W')/2;
Lt = diag(sum(W,2))-W;
Lt = Lt/norm(Lt,'fro');
%compute L
L = blkdiag(Ls,Lt);
L = L/norm(L,'fro');

% initial G
svmmodel = train(double(Ys), sparse(double(X(:,1:ns)')),'-s 1 -B 1.0 -q');
[Yt0,~,~] = predict(Yt, sparse(X(:,ns+1:end)'), svmmodel,'-q');

Acc = length(find(Yt == Yt0))/nt;
Gt = full(sparse(1:nt,Yt0,1));
if size(Gt,2) < C
    Gt = [Gt,zeros(nt,C-size(Gt,2))];
end
G = [zeros(ns,C);Gt];

for i = 1:T
    
    %update Q
    U = eye(C)/(alpha*(G'*G) + eye(C));
    Ess = Es'* Es;
    Esu = Ess * U;
    Ett = Gt'*Gt;
    Etu = Ett*U;
    tr11 = trace(Ess*Ess')-2*trace(Ess*Esu)+trace(Esu*Esu);
    tr12 = alpha*alpha*trace(Esu*Etu);
    tr22 = alpha*alpha*(nt-2*alpha*trace(Etu)+alpha*alpha*trace(Etu*Etu));
    normR = sqrt(tr11 + 2*tr12+tr22);
    Xe = Xss*Es;
    Xg = Xtt*Gt;
    R = (Xe*Xe'-Xe*U*Xe' - alpha*Xe*U*Xg' - alpha*Xg*U*Xe'+ alpha*(Xtt*Xtt')-alpha*alpha*Xg*U*Xg')/normR;
    [Q,~] = eigs(R + gamma*X*L*X' + beta*eye(m),T1,d,'SM');
    
    %update F
    Z = Q'*X;
    Z = Z-mean(Z,2);
    Z = Z*diag((1./sqrt(sum(Z.^2))));
    F = (Z*E + alpha*Z*G)*U;
    
    %update G
    Zt = Z(:,ns+1:end);
    for j = 1:nt
        Yt0(j) = searchBestIndicator(Zt(:,j),F,C);
    end
    Gt = full(sparse(1:nt,Yt0,1));
    if size(Gt,2) < C
        Gt = [Gt,zeros(nt,C-size(Gt,2))];
    end
    G = [zeros(ns,C);Gt];
    
    %update Lt
    Lt = computeLt(Zt,k,r);
    %update L
    L = blkdiag(Ls,Lt);
    L = L/norm(L,'fro');
    %compute accuracy
    Acc = length(find(Yt == Yt0))/nt;
    for cc = 1:12
        ind = find(Yt==cc);
        Yt0c = Yt0(ind);
        fprintf('%.1f ', length(find(Yt0c==cc))/length(ind)*100);
    end
    fprintf('\n');
end
end
