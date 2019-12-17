addpath('liblinear-2.30/windows/')
options.d = 100;
options.alpha = 0.1;
options.beta = 0.01;
options.gamma = 5.0;
options.T = 10;
Str = {'Art','Clipart','Product','RealWorld'};
avarage_acc = 0.0;
for i = 1:4
    for j = 1:4
        if i == j
            continue
        end
        src = char(Str{i});
        tgt = char(Str{j});
        load(['../data/Office-Home/OfficeHome-' src  '-resnet50-noft.mat']);
        fts = double(resnet50_features);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = double((zscore(fts,1))');
        Ys = double(labels') + 1;

        load(['../data/Office-Home/OfficeHome-' tgt  '-resnet50-noft.mat']);
        fts = double(resnet50_features);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = double((zscore(fts,1))');
        Yt = double(labels') + 1;
        Acc = CMMS(Xs,Xt,Ys,Yt,options);
        fprintf('%s --> %s: %.1f%% accuracy\n', src, tgt, Acc * 100);
        avarage_acc = avarage_acc + Acc*100;
    end
end
acc = avarage_acc/12;
fprintf('CMMS mean accuracy is: %.1f%%\n',acc);
