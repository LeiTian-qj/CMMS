addpath('liblinear-2.30/windows/')
options.d = 100;
options.alpha = 0.2;
options.beta = 0.05;
options.gamma = 5.0;
options.T = 10;
avarage_acc = 0.0;
Str = {'mnist','usps'};
for i = 1: 2
    src = Str{i};
    tgt = Str{3-i};
    data = csvread(['../data/' src '_' src '.csv']);
    fts = double(data(:,1:2048));
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xs = double((zscore(fts,1))');
    Ys = data(:,2049)+1;
    data = csvread(['../data/' src '_' tgt '.csv']);
    fts = double(data(:,1:2048));
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xt = double((zscore(fts,1))');
    Yt = data(:,2049)+1;
    Acc = KMDA(Xs,Xt,Ys,Yt,options);
    fprintf('%s --> %s: %.1f%% accuracy\n', src, tgt, Acc * 100);
    avarage_acc = avarage_acc + Acc*100;
end
acc = avarage_acc/2;
fprintf('Mean accuracy is: %.1f%%\n',acc);
