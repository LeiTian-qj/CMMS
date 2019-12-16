function [Acc] = CMMS(Xs,Xt,Ys,Yt,options)
%%Inputs:
%%%Xs: source data matrix, m*ns
%%%Xt: target data matrix, m*nt
%%%Ys: source label matrix, ns*1
%%%Yt: target label matrix, nt*1 (only used for testing accuracy)
%%%options: algorithm options
%%%%options.d: subspace dimensionality (default d = 100)
%%%%options.alpha: parameter for K-means clustering
%%%%options.belta: regularization parameter for projection matrix
%%%%options.gamma: parameter for data structure information (default gamma = 5.0)
%%%%options.k: neighborhood size (default k = 10)

%%outputs:
%%%Acc: Final accuracy value
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
X = X*diag((1./sqrt(sum(X.^2))));
C = length(unique(Ys));
n = ns + nt;
H = eye(n)-1/(n)*ones(n,n); %centering matrix
Es = zeros(ns,C);
for c = reshape(unique(Ys),1,C)
    Es(find(Ys==c),c) = 1/length(find(Ys==c));
end
E = [Es;zeros(nt,C)];
V = blkdiag(zeros(ns,ns),eye(nt));

%%compute Ls
Ls = computeLs(Ys,C);

%%initialize Lt
distXt = L2_distance_1(X(:,ns+1:end),X(:,ns+1:end));
[distX1, idx] = sort(distXt,2);
S = zeros(nt,nt);
rr = zeros(nt,1);
for i = 1:nt
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
r = mean(rr);
S = (S+S')/2;
Lt = diag(sum(S,2))-S;
Lt = Lt/norm(Lt,'fro');

%compute L
L = blkdiag(Ls,Lt);
L = L/norm(L,'fro');

% initialize Gt
svmmodel = train(double(Ys), sparse(double(X(:,1:ns)')),'-s 1 -B 1.0 -q');
[Yt0,~,~] = predict(Yt, sparse(X(:,ns+1:end)'), svmmodel,'-q');
%Acc = length(find(Yt == Yt0))/nt;
Gt = full(sparse(1:nt,Yt0,1));
if size(Gt,2) < C
    Gt = [Gt,zeros(nt,C-size(Gt,2))];
end

% initialize G
G = [zeros(ns,C);Gt];

for i = 1:T
    
    %update P
    U = inv(alpha*(G'*G) + eye(C));
    R = E*E'-E*U*(E + alpha*V*G)' + alpha*(V*V')-alpha*V*G*U*(E+alpha*V*G)';
    R = R/norm(R,'fro');
    [P,~] = eigs(X*(R + gamma*L)*X' + beta*eye(m),X*H*X',d,'SM');
    
    %update F
    Z = P'*X;
    Z = Z-mean(Z,2);
    Z = Z*diag((1./sqrt(sum(Z.^2))));
    F = (Z*E + alpha*Z*V*G)*U;
    
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
    
    %compute Acc
    Acc = length(find(Yt == Yt0))/nt;
end
end
