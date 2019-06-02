clear;clc;
% data = xlsread('FT06.xlsx');  %55
% data = xlsread('LA01.xlsx');  %666
% data = xlsread('FT10.xlsx');  %930
data = xlsread('FT35.xlsx');  %1888
order = data(:, 1:2:end);
order = order + 1;
time = data(:, 2:2:end);

% [fit, sol] = JSP_GA(time, order, 0.95, 0.6, 10000);
% min(fit)

record = zeros(10, 2);
for i = 1: 10
tic
% [fit, sol] = JSP_GA(time, order, 0.95, 0.05, 50);
% [fit, sol] = JSP_GA(time, order, 0.95, 0.05, 100);
[fit, sol] = JSP_GA(time, order, 0.95, 0.05, 6000);
record(i, 1) = min(fit);
min(fit)
toc
record(i, 2) = toc;
end






% % %%
% clear;clc;
% data = xlsread('FT06.xlsx');
% % data = xlsread('FT10.xlsx');
% % data = xlsread('LA01.xlsx');
% order = data(:, 1:2:end);
% order = order + 1;
% time = data(:, 2:2:end);
% 
% tic
% parpool(2)
% spmd
%   % build magic squares in parallel
%  [fit, sol] = JSP_GA(time, order, 0.95, 0.05, 1000);
% end
% min(fit{1})
% min(fit{2})
% % min(fit{3})
% delete(gcp)
% toc




