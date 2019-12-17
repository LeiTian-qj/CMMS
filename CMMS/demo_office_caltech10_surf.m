addpath('liblinear-2.30/windows/')
options.d = 100;
options.alpha = 0.1;
options.beta = 0.2;
options.gamma = 5.0;
options.T = 10;
str_domains = {'amazon','Caltech10','dslr','webcam'};
avarage_acc = 0.0;
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load(['../data/Office-Caltech10/' src '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = double((zscore(fts,1))'); 
        Ys = labels;
        
        load(['../data/Office-Caltech10/' tgt '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = double((zscore(fts,1))'); 
        Yt = labels;
        Acc = CMMS(Xs,Xt,Ys,Yt,options);
        fprintf('%s --> %s: %.1f%% accuracy\n', src, tgt, Acc * 100);
        avarage_acc = avarage_acc + Acc*100;
    end
end
acc = avarage_acc/12;
fprintf('CMMS mean accuracy is: %.1f%%\n',acc);
