options.d = 100;
options.alpha = 0.1;
options.beta = 0.01;
options.gamma = 5.0;
options.T = 10;
data = csvread(['../data/train_resnet50.csv']);
fts = double(data(:,1:2048));
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = double((zscore(fts,1))'); 
Ys = data(:,2049) + 1; 
data = csvread(['../data/validation_resnet50.csv']);
fts = double(data(:,1:2048));
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = double((zscore(fts,1))'); 
Yt = data(:,2049) + 1;
C = length(unique(Yt));
Acc = CMMS_lg(Xs,Xt,Ys,Yt,options);


