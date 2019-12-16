addpath('liblinear-2.30/windows/')
options.d = 100;
options.alpha = 0.1;
options.beta = 0.05;
options.gamma = 5.0;
options.T = 15;
src_domain = {'MSRC','VOC'};
avarage_acc = 0.0;
for i = 1 : 2
    load(['../data/MSRC-VOC2007/' src_domain{1} '_vs_' src_domain{2} '.mat']);
    src = src_domain{i};
    tgt = src_domain{3-i};
    if i == 2
        u = X_src;
        v = Y_src;
        X_src = X_tar;
        X_tar = u;
        Y_src = Y_tar;
        Y_tar = v;
    end    
    X_src = full(X_src ./ repmat(sum(X_src,2),1,size(X_src,2))); 
    Xs = double((zscore(X_src,1))'); 
    Ys = full(Y_src);
    X_tar = full(X_tar ./ repmat(sum(X_tar,2),1,size(X_tar,2))); 
    Xt = double((zscore(X_tar,1))'); 
    Yt = full(Y_tar);
    Acc = CMMS(Xs,Xt,Ys,Yt,options);
    avarage_acc = avarage_acc + Acc*100;
    fprintf('%s --> %s: %.1f%% accuracy\n', src,  tgt, Acc * 100);
end
acc = avarage_acc/2;
fprintf('CMMS mean accuracy is: %.1f%%\n',acc);