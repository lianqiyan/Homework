clear;clc;
t_dat = importdata('Training-set.csv');
t_dat = t_dat.data;
tr_dat = t_dat(:, 2:4);
tr_la = t_dat(:, 5);
p_dat =  importdata('Testing-set-label.csv');
p_dat = p_dat.data;
pr_dat = p_dat(:, 2:4);
pr_la = p_dat(:, 5);

hold on
grid on
scatter3(tr_dat(tr_la==1, 1), tr_dat(tr_la==1, 2),tr_dat(tr_la==1, 3), 'p')
scatter3(tr_dat(tr_la==0, 1), tr_dat(tr_la==0, 2), tr_dat(tr_la==0, 2), 'r')
scatter3(pr_dat(:,1), pr_dat(:,2), pr_dat(:,3), 'filled','k')
xlabel('x1')
ylabel('x2')
zlabel('x3')


% chebychev
% correlation
% cityblock
% mahalanobis
% minkowski
% seuclidean
% euclidean
%  k = 1:10;
%  record = zeros(10, 4);
%  for i = k
% mdl = fitcknn(tr_dat,tr_la,'NumNeighbors',i,'Distance', 'euclidean');
% [label,score,cost] = predict(mdl,tr_dat);
%     record(i, 1) = sum(label==tr_la)/length(label);
%     [label,scores] = predict(mdl, pr_dat);
%     record(i, 2) = sum(label==pr_la)/length(label);
%     p = classperf(pr_la,label);
%    record(i, 3) = p.PositivePredictiveValue;
%    record(i, 4) = p.NegativePredictiveValue;
%  end


%  [b,dev,stats] = glmfit(tr_dat,tr_la,'binomial','logit'); % Logistic regression
%   p_la = round(glmval(b, pr_dat,'logit'));
% accrucy = sum(p_la==pr_la)/length(p_la)


% mdl = fitcdiscr(tr_dat,tr_la);
% label = predict(mdl,pr_dat);
% accrucy = sum(label==pr_la)/length(label)

% trainFcn = 'traingda';
% net = feedforwardnet(100, trainFcn);
% net = train(net,tr_dat',tr_la');
% label = round(net(pr_dat'))';
% accrucy = sum(label==pr_la)/length(label)
% perf = perform(net,y,t)

% gaussian
%  rbf
% 

% sv = cl.SupportVectors;
% gscatter(tr_dat(:,1),tr_dat(:,2),tr_dat(:,3), tr_la)
% hold on
% plot3(sv(:,1),sv(:,2),sv(:,3),'ko','MarkerSize',10)

% cl = fitcsvm(tr_dat,tr_la , 'KernelFunction', 'rbf', 'KernelScale', 0.4);
% ks = 0.9:-0.1:0.1;
% record = zeros(length(ks), 4);
% for i = 1:length(ks)
%     cl = fitcsvm(tr_dat,tr_la,'KernelFunction','gaussian','KernelScale', ks(i));
%     [label,~] = predict(cl, tr_dat);
%     record(i, 1) = sum(label==tr_la)/length(label);
%     [label,scores] = predict(cl, pr_dat);
%     record(i, 2) = sum(label==pr_la)/length(label);
%     p = classperf(pr_la,label);
%    record(i, 3) = p.PositivePredictiveValue;
%    record(i, 4) = p.NegativePredictiveValue;
% end